import numpy as np
import torch
import onnxruntime


"""
Code is adapted from silero-vad v6: https://github.com/snakers4/silero-vad
"""


class VADOnnxWrapper:
    """ONNX Runtime wrapper for Silero VAD model."""

    def __init__(self, path, force_onnx_cpu=False):

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        print(onnxruntime.get_available_providers())
        if force_onnx_cpu and 'CPUExecutionProvider' in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'], sess_options=opts)
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        self.reset_states()
        self.sample_rates = [16000]

    def _validate_input(self, x, sr: int):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:,::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size=1):
        self._state = torch.zeros((2, batch_size, 128)).float()
        self._context = torch.zeros(0)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int):

        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if x.shape[-1] != num_samples:
            raise ValueError(f"Provided number of samples is {x.shape[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)")

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = torch.zeros(batch_size, context_size)

        x = torch.cat([self._context, x], dim=1)
        if sr in [8000, 16000]:
            ort_inputs = {'input': x.numpy(), 'state': self._state.numpy(), 'sr': np.array(sr, dtype='int64')}
            ort_outs = self.session.run(None, ort_inputs)
            out, state = ort_outs
            self._state = torch.from_numpy(state)
        else:
            raise ValueError()

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        out = torch.from_numpy(out)
        return out


def load_silero_vad(model_path: str, force_onnx_cpu=False)-> VADOnnxWrapper:
    try:
        model = VADOnnxWrapper(path=model_path, force_onnx_cpu=force_onnx_cpu)
    except ImportError:
        raise ImportError(
            "ONNX runtime not available. Install with: pip install onnxruntime\n"
            "Or use JIT model by setting onnx=False"
        )
    return model


class VADIterator:
    """
    Voice Activity Detection iterator for streaming audio.

    This is the Silero VAD v6 implementation.
    """

    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30
                 ):

        """
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit/.onnx silero VAD model

        threshold: float (default - 0.5)
        Порог речевого сигнала. VAD Silero выводит вероятности речевого сигнала для каждого аудиофрагмента.
        Вероятности ВЫШЕ этого значения считаются РЕЧЬЮ.
        Этот параметр лучше настраивать для каждого набора данных отдельно, но «ленивое» значение 0,5 достаточно для большинства наборов данных.

        sampling_rate: int (default - 16000)
        В настоящее время модели VAD Silero поддерживают частоты дискретизации 8000 и 16000.

        min_silence_duration_ms: int (по умолчанию - 100 миллисекунд)
        В конце каждого фрагмента речевого сигнала необходимо дождаться min_silence_duration_ms перед его разделением.

        speech_pad_ms: int (по умолчанию - 30 миллисекунд)
        Конечные фрагменты речевого сигнала дополняются значением speech_pad_ms с каждой стороны.
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):

        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    @torch.no_grad()
    def __call__(self, x, return_seconds=False, time_resolution: int = 1):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)

        time_resolution: int (default - 1)
            time resolution of speech coordinates when requested as seconds
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = max(0, self.current_sample - self.speech_pad_samples - window_size_samples)
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, time_resolution)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, time_resolution)}

        return None


class RealtimeVADIterator(VADIterator):
    """
    Realtime VAD Iterator that handles variable-length audio chunks, not only exactly 512 frames at once.
    """

    def reset_states(self):
        super().reset_states()
        self.buffer = np.array([], dtype=np.float32)

    def __call__(self, x, return_seconds=False):
        self.buffer = np.append(self.buffer, x)
        ret = None
        while len(self.buffer) >= 512:
            r = super().__call__(self.buffer[:512], return_seconds=return_seconds)
            self.buffer = self.buffer[512:]
            if ret is None:
                ret = r
            elif r is not None:
                if "end" in r:
                    ret["end"] = r["end"]
                if "start" in r and "end" in ret:
                    del ret["end"]
        return ret if ret != {} else None


if __name__ == "__main__":
    import os
    from pathlib import Path

    model_path = os.path.join(Path(__file__).parent.parent, "models", "silero_vad_16k_op15.onnx") # модель silero_vad_16k_op15 заточена под 16000hz
    model = load_silero_vad(model_path=model_path, force_onnx_cpu=True)
    vad = RealtimeVADIterator(model)

    audio_buffer = np.array([0] * 512, dtype=np.float32)
    result = vad(audio_buffer)
    print(f"   512 samples: {result}")

    # test with 511 samples
    audio_buffer = np.array([0] * 511, dtype=np.float32)
    result = vad(audio_buffer)
    print(f"   511 samples: {result}")
