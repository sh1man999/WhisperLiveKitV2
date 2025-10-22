import asyncio
from typing import Any, AsyncGenerator

import numpy as np
from time import time
import logging
import traceback
from whisperlivekit.timed_objects import Silence, Line, FrontData, State, Transcript, ChangeSpeaker
from whisperlivekit.transcription_engine import TranscriptionEngine, online_diarization_factory
from whisperlivekit.silero_vad_iterator import FixedVADIterator
from whisperlivekit.results_formater import format_output
from whisperlivekit.ffmpeg_manager import FFmpegManager, FFmpegState
from whisperlivekit.whisper_streaming_custom.online_asr import OnlineASRProcessor

logger = logging.getLogger(__name__)

SENTINEL = object() # Маркер для конца потока


class AudioProcessor:
    """
    Обрабатывает аудиопотоки для транскрипции и диаризации.
    Обеспечивает обработку аудио, управление состоянием и форматирование результатов.
    """
    
    def __init__(self,
                 transcription_engine: TranscriptionEngine,
                 language: str = "auto",
                 url: str|None = None):
        """Инициализируйте аудиопроцессор с конфигурацией, моделями и состоянием."""
        self.language = language
        self.url = url
        
        # Audio processing settings
        self.args = transcription_engine.args
        self.sample_rate = 16000
        self.channels = 1
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size_sec) # Это минимальное количество сэмплов в одном чанке, которое AudioProcessor будет обрабатывать.
        self.bytes_per_sample = 2 # Почему 2? Потому что система работает со стандартным несжатым аудио-форматом PCM s16le (signed 16-bit little-endian). 16 бит — это ровно 2 байта.
        self.min_chunk_size_bytes = self.samples_per_sec * self.bytes_per_sample # Это минимальный размер чанка в байтах, который запускает обработку.
        self.max_chunk_size_bytes = (self.sample_rate * 2) * self.args.max_chunk_size_sec  # Это максимальный размер чанка в байтах, который система обработает за один раз. Это "потолок", который предотвращает переполнение и слишком большие задержки.

        # State management
        self.is_stopping = False
        self.silence = False
        self.silence_duration = 0.0
        self.tokens = []
        self.last_validated_token = 0
        self.buffer_transcription = Transcript()
        self.end_buffer = 0
        self.end_attributed_speaker = 0
        self.lock = asyncio.Lock()
        self.beg_loop = time() if self.url else 0.0 # Initialize beg_loop if URL is present
        self.sep = " "  # Default separator
        self.last_response_content = FrontData()
        self.last_detected_speaker = None
        self.speaker_languages = {}
        self.diarization_before_transcription = False

        if self.diarization_before_transcription:
            self.cumulative_pcm = []
            self.last_start = 0.0
            self.last_end = 0.0
        
        # Models and processing
        self.asr = transcription_engine.asr
        self.vac_model = transcription_engine.vac_model
        self.vac = FixedVADIterator(model=transcription_engine.vac_model, min_silence_duration_ms=1000, threshold=0.2) # threshold 0.2 or 0.5? min_silence_duration_ms?, speech_pad_ms 400?
                         
        self.ffmpeg_manager = None
        self.ffmpeg_reader_task = None
        self._ffmpeg_error = None

        if self.url:
            self.ffmpeg_manager = FFmpegManager(
                sample_rate=self.sample_rate,
                channels=self.channels,
                url=url
            )
            async def handle_ffmpeg_error(error_type: str):
                logger.error(f"FFmpeg error: {error_type}")
                self._ffmpeg_error = error_type

            self.ffmpeg_manager.on_error_callback = handle_ffmpeg_error
             
        self.transcription_queue = asyncio.Queue()
        self.diarization_queue = asyncio.Queue() if self.args.diarization else None
        self.pcm_buffer = bytearray()

        self.transcription_task = None
        self.diarization_task = None
        self.translation_task = None
        self.watchdog_task = None
        self.all_tasks_for_cleanup = []
        
        self.transcription = None
        self.diarization = None

        self.transcription = OnlineASRProcessor(transcription_engine.asr)
        self.sep = self.transcription.asr.sep
        if self.args.diarization:
            logger.info(f"Это {type(self.args.diarization)} значение {self.args.diarization}")
            self.diarization = online_diarization_factory(self.args, transcription_engine.diarization_model)

    @staticmethod
    def convert_pcm_to_float(pcm_buffer):
        """Преобразовать буфер PCM в формате s16le в нормализованный массив NumPy."""
        return np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

    @staticmethod
    def _cut_at(cumulative_pcm, cut_sec):
        cumulative_len = 0
        cut_sample = int(cut_sec * 16000)

        for ind, pcm_array in enumerate(cumulative_pcm):
            if (cumulative_len + len(pcm_array)) >= cut_sample:
                cut_chunk = cut_sample - cumulative_len
                before = np.concatenate(
                    cumulative_pcm[:ind] + [cumulative_pcm[ind][:cut_chunk]]
                )
                after = [cumulative_pcm[ind][cut_chunk:]] + cumulative_pcm[ind + 1 :]
                return before, after
            cumulative_len += len(pcm_array)
        return np.concatenate(cumulative_pcm), []


    async def get_current_state(self):
        """Получить текущее состояние."""
        async with self.lock:
            current_time = time()
            
            # Рассчитать оставшееся время
            remaining_transcription = 0
            if self.end_buffer > 0:
                remaining_transcription = max(0, round(current_time - self.beg_loop - self.end_buffer, 1))
                
            remaining_diarization = 0
            if self.tokens:
                latest_end = max(self.end_buffer, self.tokens[-1].end if self.tokens else 0)
                remaining_diarization = max(0, round(latest_end - self.end_attributed_speaker, 1))
                
            return State(
                tokens=self.tokens.copy(),
                last_validated_token=self.last_validated_token,
                buffer_transcription=self.buffer_transcription,
                end_buffer=self.end_buffer,
                end_attributed_speaker=self.end_attributed_speaker,
                remaining_time_transcription=remaining_transcription,
                remaining_time_diarization=remaining_diarization,
            )
            
    async def reset(self):
        """Сбросить все переменные состояния до начальных значений."""
        async with self.lock:
            self.tokens = []
            self.buffer_transcription = Transcript()
            self.end_buffer = self.end_attributed_speaker = 0
            self.beg_loop = time()

    async def ffmpeg_stdout_reader(self):
        """Считывает аудиоданные из stdout вывода FFmpeg и обрабатывать их в конвейере PCM."""
        beg = time()
        while True:
            try:
                if self.is_stopping:
                    logger.info("Остановка ffmpeg_stdout_reader из-за флага остановки.")
                    break

                state = await self.ffmpeg_manager.get_state() if self.ffmpeg_manager else FFmpegState.STOPPED
                if state == FFmpegState.FAILED:
                    logger.error("FFmpeg находится в состоянии FAILED, не может прочитать данные")
                    break
                elif state == FFmpegState.STOPPED:
                    logger.info("FFmpeg остановлен")
                    break
                elif state != FFmpegState.RUNNING:
                    await asyncio.sleep(0.1)
                    continue

                current_time = time()
                elapsed_time = max(0.0, current_time - beg)

                bytes_per_second = (
                    self.sample_rate * self.channels * self.bytes_per_sample
                )

                buffer_size = max(int(bytes_per_second * elapsed_time), 4096)  # dynamic read
                beg = current_time

                chunk = await self.ffmpeg_manager.read_data(buffer_size)
                if not chunk:
                    # В настоящее время данные отсутствуют.
                    logger.info(f"FFmpeg вернул пустой фрагмент (тайм-аут или конец потока)")
                    await asyncio.sleep(0.05)
                    continue

                self.pcm_buffer.extend(chunk)
                # Только тогда, когда буфер достигает порога обработки
                if len(self.pcm_buffer) >= self.min_chunk_size_bytes:
                    logger.debug(
                        f"Буфер готов к обработке: {len(self.pcm_buffer)} bytes ({len(self.pcm_buffer) / bytes_per_second:.2f}s)"
                    )

                await self.handle_pcm_data()

            except asyncio.CancelledError:
                logger.info("ffmpeg_stdout_reader отменен.")
                break
            except Exception as e:
                logger.warning(f"Исключение в ffmpeg_stdout_reader: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.2)

        logger.info("FFmpeg обработка stdout завершена. При необходимости отправляется сигнал нижестоящим процессорам.")
        if not self.diarization_before_transcription and self.transcription_queue:
            await self.transcription_queue.put(SENTINEL)
        if self.diarization:
            await self.diarization_queue.put(SENTINEL)

    async def transcription_processor(self):
        """Обработка аудиофрагментов для транскрипции."""
        cumulative_pcm_duration_stream_time = 0.0
        
        while True:
            try:
                item = await self.transcription_queue.get()
                if item is SENTINEL:
                    logger.debug("Процессор транскрипции получил сигнал. Завершение...")
                    self.transcription_queue.task_done()
                    break

                asr_internal_buffer_duration_s = len(getattr(self.transcription, 'audio_buffer', [])) / self.transcription.SAMPLING_RATE
                transcription_lag_s = max(0.0, time() - self.beg_loop - self.end_buffer)
                asr_processing_logs = f"internal_buffer={asr_internal_buffer_duration_s:.2f}s | lag={transcription_lag_s:.2f}s |"
                if type(item) is Silence:
                    asr_processing_logs += f" + Silence of = {item.duration:.2f}s"
                    if self.tokens:
                        asr_processing_logs += f" | last_end = {self.tokens[-1].end} |"
                    logger.debug(asr_processing_logs)
                    cumulative_pcm_duration_stream_time += item.duration
                    self.transcription.insert_silence(item.duration, self.tokens[-1].end if self.tokens else 0)
                    continue
                elif isinstance(item, ChangeSpeaker):
                    self.transcription.new_speaker(item)
                elif isinstance(item, np.ndarray):
                    pcm_array = item
                
                logger.debug(asr_processing_logs)
                
                duration_this_chunk = len(pcm_array) / self.sample_rate
                cumulative_pcm_duration_stream_time += duration_this_chunk
                stream_time_end_of_current_pcm = cumulative_pcm_duration_stream_time

                self.transcription.insert_audio_chunk(pcm_array, stream_time_end_of_current_pcm)
                new_tokens, current_audio_processed_upto = await asyncio.to_thread(self.transcription.process_iter,language=self.language)
                
                _buffer_transcript = self.transcription.get_buffer()
                buffer_text = _buffer_transcript.text

                if new_tokens:
                    validated_text = self.sep.join([t.text for t in new_tokens])
                    if buffer_text.startswith(validated_text):
                        _buffer_transcript.text = buffer_text[len(validated_text):].lstrip()

                candidate_end_times = [self.end_buffer]

                if new_tokens:
                    candidate_end_times.append(new_tokens[-1].end)
                
                if _buffer_transcript.end is not None:
                    candidate_end_times.append(_buffer_transcript.end)
                
                candidate_end_times.append(current_audio_processed_upto)
                
                async with self.lock:
                    self.tokens.extend(new_tokens)
                    self.buffer_transcription = _buffer_transcript
                    self.end_buffer = max(candidate_end_times)

                        
                self.transcription_queue.task_done()
                
            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'pcm_array' in locals() and pcm_array is not SENTINEL : # Check if pcm_array was assigned from queue
                    self.transcription_queue.task_done()
        
        if self.is_stopping:
            logger.info("Процессор транскрипции завершен из-за флага остановки.")
            if self.diarization_queue:
                await self.diarization_queue.put(SENTINEL)

        logger.info("Задача процессора транскрипции завершена.")


    async def diarization_processor(self, diarization_obj):
        """Обрабатывает аудиофрагменты для диаризации говорящих."""
        if self.diarization_before_transcription:
            self.current_speaker = 0
            await self.transcription_queue.put(ChangeSpeaker(speaker=self.current_speaker, start=0.0))
        while True:
            try:
                item = await self.diarization_queue.get()
                if item is SENTINEL:
                    logger.debug("Процессор диаризации получил сигнал тревоги. Завершение.")
                    self.diarization_queue.task_done()
                    break
                elif type(item) is Silence:
                    diarization_obj.insert_silence(item.duration)
                    continue
                elif isinstance(item, np.ndarray):
                    pcm_array = item
                else:
                    raise Exception('item should be pcm_array') 
                
                
                
                # Process diarization
                # https://github.com/QuentinFuxa/WhisperLiveKit/issues/251
                await diarization_obj.diarize(pcm_array)
                if self.diarization_before_transcription:
                    segments = diarization_obj.get_segments()
                    self.cumulative_pcm.append(pcm_array)
                    if segments:
                        last_segment = segments[-1]                    
                        if last_segment.speaker != self.current_speaker:
                            cut_sec = last_segment.start - self.last_end
                            to_transcript, self.cumulative_pcm = self._cut_at(self.cumulative_pcm, cut_sec)
                            await self.transcription_queue.put(to_transcript)
                            
                            self.current_speaker = last_segment.speaker
                            await self.transcription_queue.put(ChangeSpeaker(speaker=self.current_speaker, start=last_segment.start))
                            
                            cut_sec = last_segment.end - last_segment.start
                            to_transcript, self.cumulative_pcm = self._cut_at(self.cumulative_pcm, cut_sec)
                            await self.transcription_queue.put(to_transcript)                            
                            self.last_start = last_segment.start
                            self.last_end = last_segment.end
                        else:
                            cut_sec = last_segment.end - self.last_end
                            to_transcript, self.cumulative_pcm = self._cut_at(self.cumulative_pcm, cut_sec)
                            await self.transcription_queue.put(to_transcript)
                            self.last_end = last_segment.end
                elif not self.diarization_before_transcription:           
                    async with self.lock:
                        self.tokens = diarization_obj.assign_speakers_to_tokens(
                            self.tokens,
                            use_punctuation_split=self.args.punctuation_split
                        )
                if len(self.tokens) > 0:
                    self.end_attributed_speaker = max(self.tokens[-1].end, self.end_attributed_speaker)
                self.diarization_queue.task_done()
                
            except Exception as e:
                logger.warning(f"Exception in diarization_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'pcm_array' in locals() and pcm_array is not SENTINEL:
                    self.diarization_queue.task_done()
        logger.info("Задача процессора диаризации завершена.")

    async def results_formatter(self):
        """Format результаты обработки для вывода."""
        while True:
            try:
                if self._ffmpeg_error:
                    yield FrontData(status="error", error=f"FFmpeg error: {self._ffmpeg_error}")
                    self._ffmpeg_error = None
                    await asyncio.sleep(1)
                    continue

                state = await self.get_current_state()

                lines, undiarized_text = format_output(
                    state,
                    self.silence,
                    current_time=time() - self.beg_loop,
                    args=self.args,
                    sep=self.sep,
                )
                if lines and lines[-1].speaker == -2:
                    buffer_transcription = Transcript()
                else:
                    buffer_transcription = state.buffer_transcription

                buffer_diarization = ''
                if undiarized_text:
                    buffer_diarization = self.sep.join(undiarized_text)

                    async with self.lock:
                        self.end_attributed_speaker = state.end_attributed_speaker
                
                response_status = "active_transcription"
                if not state.tokens and not buffer_transcription and not buffer_diarization:
                    response_status = "no_audio_detected"
                    lines = []
                elif not lines:
                    lines = [Line(
                        speaker=1,
                        start=state.end_buffer,
                        end=state.end_buffer
                    )]
                
                response = FrontData(
                    status=response_status,
                    lines=lines,
                    buffer_transcription=buffer_transcription.text.strip(),
                    buffer_diarization=buffer_diarization,
                    remaining_time_transcription=state.remaining_time_transcription,
                    remaining_time_diarization=state.remaining_time_diarization if self.args.diarization else 0
                )
                                
                should_push = (response != self.last_response_content)
                if should_push and (lines or buffer_transcription or buffer_diarization or response_status == "no_audio_detected"):
                    yield response
                    self.last_response_content = response
                
                # Проверка на наличие условий завершения
                if self.is_stopping:
                    all_processors_done = True
                    if self.args.transcription and self.transcription_task and not self.transcription_task.done():
                        all_processors_done = False
                    if self.args.diarization and self.diarization_task and not self.diarization_task.done():
                        all_processors_done = False
                    
                    if all_processors_done:
                        logger.info("results_formater: Все вышестоящие процессоры завершены и находятся в состоянии остановки. Завершение.")
                        return
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.warning(f"Exception in results_formatter: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.5)
        
    async def create_tasks(self)-> AsyncGenerator[FrontData, Any]:
        """Create and start processing tasks."""
        self.all_tasks_for_cleanup = []
        processing_tasks_for_watchdog = []

        # If using FFmpeg (non-PCM input), start it and spawn stdout reader
        if self.url:
            success = await self.ffmpeg_manager.start()
            if not success:
                logger.error("Не удалось запустить менеджер FFmpeg")
                async def error_generator():
                    yield FrontData(
                        status="error",
                        error="Не удалось запустить FFmpeg. Проверьте, установлен ли FFmpeg."
                    )
                return error_generator()
            self.ffmpeg_reader_task = asyncio.create_task(self.ffmpeg_stdout_reader())
            self.all_tasks_for_cleanup.append(self.ffmpeg_reader_task)
            processing_tasks_for_watchdog.append(self.ffmpeg_reader_task)

        if self.transcription:
            self.transcription_task = asyncio.create_task(self.transcription_processor())
            self.all_tasks_for_cleanup.append(self.transcription_task)
            processing_tasks_for_watchdog.append(self.transcription_task)
            
        if self.diarization:
            self.diarization_task = asyncio.create_task(self.diarization_processor(self.diarization))
            self.all_tasks_for_cleanup.append(self.diarization_task)
            processing_tasks_for_watchdog.append(self.diarization_task)
        
        # Мониторинг общего состояния системы
        self.watchdog_task = asyncio.create_task(self.watchdog(processing_tasks_for_watchdog))
        self.all_tasks_for_cleanup.append(self.watchdog_task)
        
        return self.results_formatter()

    async def watchdog(self, tasks_to_monitor):
        """Monitors the health of critical processing tasks."""
        while True:
            try:
                await asyncio.sleep(10)
                
                for i, task in enumerate(tasks_to_monitor):
                    if task.done():
                        exc = task.exception()
                        task_name = task.get_name() if hasattr(task, 'get_name') else f"Monitored Task {i}"
                        if exc:
                            logger.error(f"{task_name} неожиданно завершено с исключением: {exc}")
                        else:
                            logger.info(f"{task_name} завершено нормально.")
                    
            except asyncio.CancelledError:
                logger.info("Watchdog task отменено.")
                break
            except Exception as e:
                logger.error(f"Ошибка в watchdog task: {e}", exc_info=True)
        
    async def cleanup(self):
        """Clean up resources when processing is complete."""
        logger.info("Начинается очистка ресурсов AudioProcessor.")
        self.is_stopping = True
        for task in self.all_tasks_for_cleanup:
            if task and not task.done():
                task.cancel()
            
        created_tasks = [t for t in self.all_tasks_for_cleanup if t]
        if created_tasks:
            await asyncio.gather(*created_tasks, return_exceptions=True)
        logger.info("Все задачи по обработке отменены или завершены.")

        if self.ffmpeg_manager:
            try:
                await self.ffmpeg_manager.stop()
                logger.info("Менеджер FFmpeg остановлен.")
            except Exception as e:
                logger.warning(f"Ошибка остановки менеджера FFmpeg: {e}")
        if self.diarization:
            self.diarization.close()
        logger.info("Очистка AudioProcessor завершена.")


    async def process_audio(self, message):
        """Обработка входящих аудиоданных."""

        if not self.beg_loop:
            self.beg_loop = time()

        if not message:
            logger.info("Получено пустое звуковое сообщение, инициируется последовательность остановки.")
            self.is_stopping = True
             
            if self.transcription_queue:
                await self.transcription_queue.put(SENTINEL)

            if self.ffmpeg_manager:
                await self.ffmpeg_manager.stop()

            return

        if self.is_stopping:
            logger.warning("AudioProcessor останавливается. Игнорирует входящий звук.")
            return

        self.pcm_buffer.extend(message)
        await self.handle_pcm_data()

    async def handle_pcm_data(self):
        # Процесс, когда достаточно данных
        buffer_duration = len(self.pcm_buffer) / self.min_chunk_size_bytes
        required_duration = self.min_chunk_size_bytes / self.min_chunk_size_bytes  # Always 1.0

        # Если в нашем буфере pcm_buffer накоплено меньше, чем self.min_chunk_size_bytes байт т.е. меньше (например 0.5) сек аудио, то ничего не делаем и ждем, пока данных не накопится больше
        if len(self.pcm_buffer) < self.min_chunk_size_bytes:
            logger.debug(f"Буфер: {buffer_duration:.2f}s / {required_duration:.2f}s - ждем больше данных")
            return

        if len(self.pcm_buffer) > self.max_chunk_size_bytes:
            logger.warning(
                f"Аудиобуфер слишком большой: {len(self.pcm_buffer) / self.min_chunk_size_bytes:.2f}s. "
                f"Рассмотрите возможность использования модели меньшего размера."
            )

        """
        # Если в буфере накопилось очень много данных (например, 10 секунд), не пытаться обработать их все сразу.
        # Взять на обработку только (например 5 секунд), а остальное оставить в буфере до следующего раза
        """
        chunk_size = min(len(self.pcm_buffer), self.max_chunk_size_bytes)
        aligned_chunk_size = (chunk_size // self.bytes_per_sample) * self.bytes_per_sample
        
        if aligned_chunk_size == 0:
            return
        pcm_array = self.convert_pcm_to_float(self.pcm_buffer[:aligned_chunk_size])
        self.pcm_buffer = self.pcm_buffer[aligned_chunk_size:]

        res = None
        end_of_audio = False
        silence_buffer = None

        if self.args.vac:
            res = self.vac(pcm_array)

        if res is not None:
            if res.get("end", 0) > res.get("start", 0):
                end_of_audio = True
            elif self.silence: #end of silence
                self.silence = False
                silence_buffer = Silence(duration=time() - self.start_silence)

        if silence_buffer:
            if not self.diarization_before_transcription and self.transcription_queue:
                await self.transcription_queue.put(silence_buffer)
            if self.args.diarization and self.diarization_queue:
                await self.diarization_queue.put(silence_buffer)

        if not self.silence:
            if not self.diarization_before_transcription and self.transcription_queue:
                await self.transcription_queue.put(pcm_array.copy())

            if self.args.diarization and self.diarization_queue:
                await self.diarization_queue.put(pcm_array.copy())

            self.silence_duration = 0.0

            if end_of_audio:
                self.silence = True
                self.start_silence = time()

        if not self.args.transcription and not self.args.diarization:
            await asyncio.sleep(0.1)
