import re
import sys
import logging
from typing import List, Union
import numpy as np
from faster_whisper.transcribe import Segment

from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.whisper_streaming_custom.constants.bad_words import BAD_WORDS
from whisperlivekit.whisper_streaming_custom.constants.skip_words import SKIP_WORDS
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class FasterWhisperASR:
    """Uses faster-whisper as the backend."""
    sep = "" # join transcribe words with this character (" " for whisper_timestamped,
    # "" for faster-whisper because it emits the spaces when needed)

    def __init__(
        self,
        lan,
        model_size=None,
        cache_dir=None,
        model_dir=None,
        logfile=sys.stderr,
        device="auto",
        device_index: Union[int, List[int]] = 0,
        cpu_threads: int = 0,
        num_workers: int = 1,
        compute_type: str = "float16",
    ):
        self.logfile = logfile
        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan

        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. "
                         f"model_size and cache_dir parameters are not used.")
            model_size_or_path = model_dir
        elif model_size is not None:
            model_size_or_path = model_size
        else:
            raise ValueError("Either model_size or model_dir must be set")

        self.model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
            device_index=device_index,
            cpu_threads=cpu_threads,
            num_workers=num_workers
        )

    def with_offset(self, offset: float) -> ASRToken:
        # This method is kept for compatibility (typically you will use ASRToken.with_offset)
        return ASRToken(self.start + offset, self.end + offset, self.text)


    def _filter_bad_segments(self, segments: List[Segment]) -> List[Segment]:
        new_segments = []

        for segment in segments:
            segment_detail = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"

            if any(re.search(bad_pattern, segment.text, re.IGNORECASE) for bad_pattern in BAD_WORDS):
                logging.info(f"Найдено bad слово/паттерн в сегменте: {segment_detail}")
                continue

            if any(re.fullmatch(skip_pattern, segment.text, re.IGNORECASE) for skip_pattern in SKIP_WORDS):
                logging.info(f"Найден skip паттерн (полное совпадение) в сегменте: {segment_detail}")
                continue

            new_segments.append(segment)

        return new_segments

    def transcribe(self, audio: np.ndarray, init_prompt: str = "", language: str = None) -> list:
        lang_to_use = language if language and language != "auto" else self.original_language
        segments, info = self.model.transcribe(
            audio,
            language=lang_to_use,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return self._filter_bad_segments(list(segments))

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.no_speech_prob > 0.9:
                continue

            for word in segment.words:
                token = ASRToken(word.start, word.end, word.word, probability=word.probability)
                tokens.append(token)
        return tokens

    def segments_end_ts(self, segments) -> List[float]:
        return [segment.end for segment in segments]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def __repr__(self):
        return (
            f"ASRToken(start={self.start:.2f}, end={self.end:.2f}, text={self.text!r})"
        )