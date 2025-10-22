#!/usr/bin/env python3
"""
Скрипт для тестирования backend_factory с измерением времени распознавания.
Использование: python scripts/backend_t.py --audio path/to/audio.wav
"""

import argparse
import logging
import time
from pathlib import Path
import sys

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import librosa

from whisperlivekit.whisper_streaming_custom.whisper_online import backend_factory
from whisperlivekit.whisper_streaming_custom.online_asr import OnlineASRProcessor
from src.backend.entrypoint.config import create_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_audio_file(
    audio_path: str,
    model_size: str = "tiny",
    language: str = "ru",
    chunk_size_sec: float = 1.0,
    device: str = "auto",
    compute_type: str = "float16",
    beam_size: int = 5,
    buffer_trimming: str = "segment",
    buffer_trimming_sec: int = 15,
    model_cache_dir: str = None,
    device_index: int | list[int] = 0,
    cpu_threads: int = 0,
    num_workers: int = 1,
) -> dict:
    """
    Обрабатывает аудио файл и возвращает метрики производительности.

    Args:
        audio_path: Путь к аудио файлу
        model_size: Размер модели Whisper (tiny, base, small, medium, large-v2, large-v3)
        language: Язык распознавания
        chunk_size_sec: Размер чанка в секундах
        device: Устройство для инференса (auto, cpu, cuda)
        compute_type: Тип вычислений (float16, int8, etc.)
        beam_size: Размер beam search
        buffer_trimming: Тип обрезки буфера (segment, sentence)
        buffer_trimming_sec: Максимальный размер буфера в секундах
        model_cache_dir: Директория для кэша моделей
        device_index: Индекс CUDA устройства
        cpu_threads: Количество потоков CPU
        num_workers: Количество воркеров

    Returns:
        Словарь с метриками производительности
    """

    # Загружаем аудио файл
    logger.info(f"Загрузка аудио файла: {audio_path}")
    audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    duration_sec = len(audio) / sample_rate
    logger.info(f"Аудио загружено: {duration_sec:.2f}s, sample_rate={sample_rate}Hz")

    # Создаем backend через factory
    logger.info("Инициализация backend_factory...")
    init_start = time.time()

    asr_backend = backend_factory(
        lan=language,
        model_size=model_size,
        model_cache_dir=model_cache_dir,
        buffer_trimming=buffer_trimming,
        buffer_trimming_sec=buffer_trimming_sec,
        confidence_validation=False,
        warmup_file=None,
        device=device,
        device_index=device_index,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        compute_type=compute_type,
        beam_size=beam_size,
        model_dir=None
    )

    init_time = time.time() - init_start
    logger.info(f"Backend инициализирован за {init_time:.2f}s")

    # Создаем онлайн процессор
    online_processor = OnlineASRProcessor(asr=asr_backend)

    # Получаем разделитель
    sep = asr_backend.sep if hasattr(asr_backend, 'sep') else " "

    # Обрабатываем аудио чанками
    logger.info(f"Начало обработки аудио (chunk_size={chunk_size_sec}s)...")
    chunk_size_samples = int(sample_rate * chunk_size_sec)

    process_start = time.time()
    first_token_time = None
    num_chunks = 0
    all_committed_tokens = []

    for i in range(0, len(audio), chunk_size_samples):
        chunk = audio[i:i + chunk_size_samples]

        if len(chunk) == 0:
            continue

        # Отправляем чанк в процессор
        online_processor.insert_audio_chunk(chunk)

        # Получаем результат
        new_tokens, current_audio_processed_upto = online_processor.process_iter(language=language)

        # Фиксируем время первого токена
        if first_token_time is None and new_tokens:
            first_token_time = time.time() - process_start
            logger.info(f"Первый токен получен за {first_token_time:.3f}s")

        num_chunks += 1

        # Накапливаем committed токены
        if new_tokens:
            all_committed_tokens.extend(new_tokens)
            committed_text = sep.join([t.text for t in new_tokens])
            logger.debug(f"Chunk {num_chunks}: {committed_text}")

    # Финализируем обработку
    remaining_tokens, final_processed_upto = online_processor.finish()

    # Добавляем оставшиеся токены
    if remaining_tokens:
        all_committed_tokens.extend(remaining_tokens)

    process_time = time.time() - process_start

    # Получаем финальный текст из всех токенов используя разделитель из asr
    final_text = sep.join([t.text for t in all_committed_tokens])

    # Собираем метрики
    metrics = {
        "audio_duration_sec": duration_sec,
        "init_time_sec": init_time,
        "process_time_sec": process_time,
        "total_time_sec": init_time + process_time,
        "first_token_latency_sec": first_token_time,
        "num_chunks": num_chunks,
        "chunk_size_sec": chunk_size_sec,
        "rtf": process_time / duration_sec if duration_sec > 0 else None,
        "transcription": final_text,
    }

    return metrics


def print_metrics(metrics: dict):
    """Красиво выводит метрики производительности."""
    print("\n" + "="*60)
    print("МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*60)
    print(f"Длительность аудио:          {metrics['audio_duration_sec']:.2f}s")
    print(f"Время инициализации модели:  {metrics['init_time_sec']:.2f}s")
    print(f"Время обработки:             {metrics['process_time_sec']:.2f}s")
    print(f"Общее время:                 {metrics['total_time_sec']:.2f}s")
    print(f"Первый токен (latency):      {metrics['first_token_latency_sec']:.3f}s" if metrics['first_token_latency_sec'] else "N/A")
    print(f"Количество чанков:           {metrics['num_chunks']}")
    print(f"Размер чанка:                {metrics['chunk_size_sec']:.2f}s")
    print(f"Real-Time Factor (RTF):      {metrics['rtf']:.3f}" if metrics['rtf'] else "N/A")
    print(f"Скорость:                    {1/metrics['rtf']:.2f}x realtime" if metrics['rtf'] else "N/A")
    print("="*60)
    print("\nТРАНСКРИПЦИЯ:")
    print("-"*60)
    print(metrics['transcription'])
    print("-"*60 + "\n")


def main():
    # Загружаем конфигурацию из переменных окружения
    try:
        config = create_config()
        logger.info("Конфигурация загружена из переменных окружения")
    except Exception as e:
        logger.warning(f"Не удалось загрузить конфигурацию из env: {e}")
        logger.warning("Используем значения по умолчанию")
        config = None

    parser = argparse.ArgumentParser(
        description="Тестирование backend_factory с измерением производительности"
    )

    # Обязательные аргументы
    parser.add_argument(
        "--audio",
        type=str,
        default=str(Path(__file__).parent / "assets" / "test.flac"),
        help="Путь к аудио файлу для тестирования"
    )

    # Параметры модели (используем значения из конфига если он есть)
    parser.add_argument(
        "--model_size",
        type=str,
        default=config.whisper.model_id if config else "tiny",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help=f"Размер модели Whisper (default: {config.whisper.model_id if config else 'tiny'})"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="ru",
        help="Язык распознавания (ru, en, etc.)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=config.whisper.device if config else "auto",
        help=f"Устройство (auto, cpu, cuda) (default: {config.whisper.device if config else 'auto'})"
    )

    parser.add_argument(
        "--compute_type",
        type=str,
        default=config.whisper.compute_type if config else "float16",
        help=f"Тип вычислений (float16, int8, etc.) (default: {config.whisper.compute_type if config else 'float16'})"
    )

    parser.add_argument(
        "--beam_size",
        type=int,
        default=config.whisper.beam_size if config else 5,
        help=f"Размер beam search (default: {config.whisper.beam_size if config else 5})"
    )

    # Параметры обработки
    parser.add_argument(
        "--chunk_size",
        type=float,
        default=config.min_chunk_size_sec if config else 1.0,
        help=f"Размер чанка в секундах (default: {config.min_chunk_size_sec if config else 1.0})"
    )

    parser.add_argument(
        "--buffer_trimming",
        type=str,
        default=config.buffer_trimming if config else "segment",
        choices=["segment", "sentence"],
        help=f"Тип обрезки буфера (default: {config.buffer_trimming if config else 'segment'})"
    )

    parser.add_argument(
        "--buffer_trimming_sec",
        type=int,
        default=config.buffer_trimming_sec if config else 15,
        help=f"Максимальный размер буфера в секундах (default: {config.buffer_trimming_sec if config else 15})"
    )

    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=config.whisper.models_path if config else None,
        help=f"Директория с моделью (default: {config.whisper.models_path if config else 'None'})"
    )


    parser.add_argument(
        "--device_index",
        type=int,
        default=config.whisper.cuda_device_index if config and isinstance(config.whisper.cuda_device_index, int) else 0,
        help=f"Индекс CUDA устройства (default: {config.whisper.cuda_device_index if config and isinstance(config.whisper.cuda_device_index, int) else 0})"
    )

    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=config.whisper.cpu_threads if config else 0,
        help=f"Количество потоков CPU (default: {config.whisper.cpu_threads if config else 0})"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=config.whisper.num_workers if config else 1,
        help=f"Количество воркеров (default: {config.whisper.num_workers if config else 1})"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Включить детальное логирование"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Проверяем существование файла
    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error(f"Аудио файл не найден: {audio_path}")
        sys.exit(1)

    # Выводим используемые параметры
    logger.info("="*60)
    logger.info("ПАРАМЕТРЫ ТЕСТИРОВАНИЯ:")
    logger.info(f"  Аудио файл: {audio_path}")
    logger.info(f"  Модель: {args.model_size}")
    logger.info(f"  Язык: {args.language}")
    logger.info(f"  Устройство: {args.device}")
    logger.info(f"  Тип вычислений: {args.compute_type}")
    logger.info(f"  Beam size: {args.beam_size}")
    logger.info(f"  Размер чанка: {args.chunk_size}s")
    logger.info(f"  Buffer trimming: {args.buffer_trimming}")
    logger.info(f"  Buffer trimming sec: {args.buffer_trimming_sec}s")
    logger.info("="*60)

    # Запускаем тестирование
    try:
        metrics = process_audio_file(
            audio_path=str(audio_path),
            model_size=args.model_size,
            language=args.language,
            chunk_size_sec=args.chunk_size,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            buffer_trimming=args.buffer_trimming,
            buffer_trimming_sec=args.buffer_trimming_sec,
            model_cache_dir=args.model_cache_dir,
            device_index=args.device_index,
            cpu_threads=args.cpu_threads,
            num_workers=args.num_workers,

        )

        print_metrics(metrics)

    except Exception as e:
        logger.error(f"Ошибка при обработке: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
