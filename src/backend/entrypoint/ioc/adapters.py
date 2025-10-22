from whisperlivekit import TranscriptionEngine

from src.backend.application.interfaces.audio_storage import AudioStorage
from dishka import Provider, Scope, provide, from_context

from src.backend.entrypoint.config import Config
from src.backend.infrastructure.adapters.local_audio_storage import LocalAudioStorage


# app dependency logic


class ConfigProvider(Provider):
    scope = Scope.APP

    config = from_context(provides=Config, scope=Scope.APP)


class StorageProvider(Provider):
    scope = Scope.APP

    audio_storage = provide(LocalAudioStorage, provides=AudioStorage)

class TranscriptionProvider(Provider):
    scope = Scope.APP

    @provide
    def transcription_engine(self, config: Config) -> TranscriptionEngine:
        return TranscriptionEngine(
            model_size=config.whisper.model_id,
            model_cache_dir=config.whisper.models_path,
            is_diarization=config.diarization,
            embedding_model_name=config.pyannote.embedding_model,
            lan="ru",  # default
            segmentation_model_name=config.pyannote.segmentation_model,
            diarization_backend="diart",
            device=config.whisper.device,
            device_index=config.whisper.cuda_device_index,
            cpu_threads=config.whisper.cpu_threads,
            num_workers=config.whisper.num_workers,
            compute_type=config.whisper.compute_type,
            split_on_punctuation_for_display=True,  # Разбивает на строки при обнаружении знаков препинаний
            beam_size=config.whisper.beam_size,
            buffer_trimming=config.buffer_trimming,  # Режим сегментации из конфига
            buffer_trimming_sec=config.buffer_trimming_sec,  # Время нарезки буфера из конфига
            min_chunk_size_sec=config.min_chunk_size_sec,
            max_chunk_size_sec=config.max_chunk_size_sec,
            vac=True, # TODO: Написать конфиг для вак
            confidence_validation=False, # Если он включен (True), метод flush будет "подтверждать" (commit) токены, даже если они не совпадают с предыдущим буфером, при условии, что их вероятность (token.probability) очень высока ( > 0.95).
        )