from dataclasses import dataclass
import logging
import os
from os import getenv

import torch
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

class BaseNeuralConfig:

    @classmethod
    def validate_device(cls, key: str, device: str, allowed_devices: set[str]) -> None:
        """Валидирует тип устройства"""
        if device not in allowed_devices:
            raise ValueError(
                f"{key} Некорректное значение device: '{device}'. " f"Допустимые значения: {allowed_devices}."
            )

    @classmethod
    def validate_cuda_devices(cls, device: str, cuda_device_index: int | list[int]) -> None:
        """Валидирует доступность CUDA устройств"""
        if device != "cuda":
            return

        if not torch.cuda.is_available():
            raise ValueError("Устройство cuda недоступно.")

        available_gpus = [(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
        logger.info(f"Доступные устройства cuda: {available_gpus}")

        indices = cuda_device_index if isinstance(cuda_device_index, list) else [cuda_device_index]
        for idx in indices:
            if idx >= len(available_gpus):
                raise ValueError(f"CUDA устройство с индексом {idx} недоступно.")

    @classmethod
    def parse_cuda_device_index(cls, cuda_device_index_str: str) -> int | list[int]:
        """
            Парсит строку с индексами CUDA-устройств.

            Поддерживает:
            - Одно число (напр., "1")
            - Список через запятую (напр., "0,1,2")
            - Пустую строку или None (возвращает 0 по умолчанию)
            """
        # 1. Обрабатываем случай по умолчанию
        if not cuda_device_index_str:
            return 0

        # Убираем лишние пробелы по краям
        processed_str = cuda_device_index_str.strip()

        # 2. Проверяем, является ли строка списком
        if "," in processed_str:
            try:
                # Отфильтровываем пустые элементы, которые могут появиться из-за "1,2,"
                indices = [int(i.strip()) for i in processed_str.split(",") if i.strip()]

                # Если после обработки остался один элемент (напр. "2,"), вернем его как число
                if len(indices) == 1:
                    return indices[0]
                return indices
            except ValueError:
                raise ValueError(f"Строка '{cuda_device_index_str}' содержит нечисловые значения в списке.")

        # 3. Если это не список, пытаемся преобразовать в одно число
        try:
            return int(processed_str)
        except ValueError:
            raise ValueError(f"Не удалось преобразовать '{cuda_device_index_str}' в число.")


@dataclass
class PyannoteConfig(BaseNeuralConfig):
    device: str
    huggingface_token: str
    segmentation_model: str
    embedding_model: str

    @classmethod
    def from_env(cls) -> "PyannoteConfig":
        device_str = getenv("PYANNOTE__DEVICE", "cuda")
        cuda_device_index_str = getenv("PYANNOTE__CUDA_DEVICE_INDEX", "0")
        huggingface_token = getenv("PYANNOTE__HUGGINGFACE_TOKEN")
        segmentation_model = getenv("PYANNOTE__SEGMENTATION_MODEL", "pyannote/segmentation")
        embedding_model = getenv("PYANNOTE__EMBEDDING_MODEL", "pyannote/embedding")

        # Используем методы родительского класса
        cls.validate_device(
            key="PYANNOTE__DEVICE", device=device_str, allowed_devices={"cpu", "cuda"}
        )
        cuda_device_index = cls.parse_cuda_device_index(cuda_device_index_str)
        cls.validate_cuda_devices(device_str, cuda_device_index)

        return cls(
            device=(
                f"{device_str}:{cuda_device_index_str}"
                if device_str == "cuda"
                else device_str
            ),
            huggingface_token=huggingface_token,
            segmentation_model=segmentation_model,
            embedding_model=embedding_model,
        )


@dataclass
class WhisperConfig(BaseNeuralConfig):
    models_path: str
    local_files_only: bool
    compute_type: str
    cuda_device_index: int | list[int]
    model_id: str
    cpu_threads: int
    num_workers: int
    device: str
    buffer_trimming: str
    buffer_trimming_sec: int
    beam_size: int

    @classmethod
    def from_env(cls) -> "WhisperConfig":
        models_path = getenv("WHISPER__MODELS_PATH")
        local_files_only = getenv("WHISPER__LOCAL_FILES_ONLY", "False") == "True"
        compute_type = getenv("WHISPER__COMPUTE_TYPE", "int8")
        cuda_device_index_str = getenv("WHISPER__CUDA_DEVICE_INDEX", "0")
        model_id = getenv("WHISPER__MODEL_ID", "large-v2")
        cpu_threads = getenv("WHISPER__CPU_THREADS", "1")
        num_workers = getenv("WHISPER__NUM_WORKERS", "1")
        device = getenv("WHISPER__DEVICE", "cuda")
        buffer_trimming = getenv("WHISPER__BUFFER_TRIMMING", "segment")
        buffer_trimming_sec = getenv("WHISPER__BUFFER_TRIMMING_SEC", "15")
        beam_size = getenv("WHISPER__BEAM_SIZE", "5")

        cls.validate_device(key="WHISPER__DEVICE", device=device, allowed_devices={"cpu", "cuda", "auto"})
        if models_path is None:
            raise ValueError("WHISPER__MODELS_PATH - переменная окружения не установлена.")
        os.makedirs(models_path, exist_ok=True)
        cuda_device_index = cls.parse_cuda_device_index(cuda_device_index_str)
        cls.validate_cuda_devices(device, cuda_device_index)

        return WhisperConfig(
            models_path=models_path,
            local_files_only=local_files_only,
            compute_type=compute_type,
            cuda_device_index=cuda_device_index,
            model_id=model_id,
            cpu_threads=int(cpu_threads),
            num_workers=int(num_workers),
            device=device,
            buffer_trimming=buffer_trimming,
            buffer_trimming_sec=int(buffer_trimming_sec),
            beam_size=int(beam_size),
        )


@dataclass
class Config:
    log_level: str
    debug: bool
    audio_storage_path: str
    whisper: WhisperConfig
    pyannote: PyannoteConfig
    log_files_path: str | None
    hostname: str
    auth_token: str
    diarization: bool

    @staticmethod
    def from_env() -> "Config":
        log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
        log_files_path = getenv("LOG_FILES_PATH")
        cuda_device_index = os.getenv("CUDA_DEVICE_INDEX", "0")
        audio_storage_path = getenv("AUDIO_STORAGE_PATH")
        debug = getenv("DEBUG", "False") == "True"
        if audio_storage_path is None:
            raise ValueError("AUDIO_STORAGE_PATH environment variable not set")
        # create dirs if not exist
        os.makedirs(audio_storage_path, exist_ok=True)
        if log_files_path is not None:
            os.makedirs(log_files_path, exist_ok=True)
        # ------------
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_level not in valid_levels:
            raise ValueError(f"⚠️  Неверный LOG_LEVEL: '{log_level}'. Доступные: {', '.join(valid_levels)}")
        whisper = WhisperConfig.from_env()
        pyannote = PyannoteConfig.from_env()
        os.environ.setdefault("CUDA_DEVICE_INDEX", cuda_device_index)
        os.environ.setdefault("HF_TOKEN", pyannote.huggingface_token)
        return Config(
            log_level=log_level,
            debug=debug,
            log_files_path=log_files_path,
            audio_storage_path=audio_storage_path,
            whisper=whisper,
            pyannote=pyannote,
            hostname=f"{os.uname().nodename}_cuda_{cuda_device_index}",
            auth_token=os.getenv("AUTH_TOKEN"),
            diarization=os.getenv("DIARIZATION", "False") == "True",
        )


def create_config() -> Config:
    return Config.from_env()
