import sys
import numpy as np
import logging
from typing import List, Tuple, Optional
from whisperlivekit.timed_objects import ASRToken, Sentence, Transcript

logger = logging.getLogger(__name__)

class HypothesisBuffer:
    """
    Буфер для хранения и обработки токенов гипотез ASR.

    Он содержит:
    - commited_in_buffer: токены, которые были подтверждены (закомичены);
    - buffer: последняя гипотеза, которая еще не закомичена;
    - new: новые токены, поступающие от распознавателя.
    """
    def __init__(self, logfile=sys.stderr, confidence_validation=False):
        self.confidence_validation = confidence_validation
        self.committed_in_buffer: List[ASRToken] = []
        self.buffer: List[ASRToken] = []
        self.new: List[ASRToken] = []
        self.last_committed_time = 0.0
        self.last_committed_word: Optional[str] = None
        self.logfile = logfile

    def insert(self, new_tokens: List[ASRToken], offset: float):
        """Вставляет новые токены (после применения временного смещения) и сравните их с
        уже зафиксированными токенами. Добавляются только токены, расширяющие зафиксированную гипотезу.
        """
        # Применить смещение к каждому токену.
        new_tokens = [token.with_offset(offset) for token in new_tokens]
        # Сохраняет только те токены, которые являются примерно «новыми».
        self.new = [token for token in new_tokens if token.start > self.last_committed_time - 0.1]

        if self.new:
            first_token = self.new[0]
            if abs(first_token.start - self.last_committed_time) < 1:
                if self.committed_in_buffer:
                    committed_len = len(self.committed_in_buffer)
                    new_len = len(self.new)
                    # Try to match 1 to 5 consecutive tokens
                    max_ngram = min(min(committed_len, new_len), 5)
                    for i in range(1, max_ngram + 1):
                        committed_ngram = " ".join(token.text for token in self.committed_in_buffer[-i:])
                        new_ngram = " ".join(token.text for token in self.new[:i])
                        if committed_ngram == new_ngram:
                            removed = []
                            for _ in range(i):
                                removed_token = self.new.pop(0)
                                removed.append(repr(removed_token))
                            logger.debug(f"Removing last {i} words: {' '.join(removed)}")
                            break

    def flush(self) -> List[ASRToken]:
        """Возвращает закомиченные чанки, определяемый как самый длинный общий префикс между предыдущей гипотезой и новыми токенами."""
        committed: List[ASRToken] = []
        while self.new:
            current_new = self.new[0]
            if self.confidence_validation and current_new.probability and current_new.probability > 0.95:
                committed.append(current_new)
                self.last_committed_word = current_new.text
                self.last_committed_time = current_new.end
                self.new.pop(0)
                self.buffer.pop(0) if self.buffer else None
            elif not self.buffer:
                break
            elif current_new.text == self.buffer[0].text:
                committed.append(current_new)
                self.last_committed_word = current_new.text
                self.last_committed_time = current_new.end
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(committed)
        return committed

    def pop_committed(self, time: float):
        """Удалить токены (с начала), которые закончились до `time`."""
        while self.committed_in_buffer and self.committed_in_buffer[0].end <= time:
            self.committed_in_buffer.pop(0)



class OnlineASRProcessor:
    """Обрабатывает входящий аудиосигнал потоковым способом, периодически обращаясь к системе автоматического распознавания речи (ASR), и использует буфер гипотез для фиксации и обрезки распознанного текста.
    Процессор поддерживает два типа обрезки буфера:
    - sentence: обрезка по границам предложений (с использованием токенизатора предложений)
    - segment: обрезка по фиксированной длительности сегментов.
        """
    SAMPLING_RATE = 16000

    def __init__(
        self,
        asr,
        logfile=sys.stderr,
    ):
        """asr: например экземпляр WhisperASR, который предоставляет метод `transcribe`, метод `ts_words` (для извлечения токенов), метод `segments_end_ts` и атрибут `sep`.
        tokenize_method: Функция, которая получает текст и возвращает список строк предложений.
        buffer_trimming: Кортеж (параметр, секунды), где параметр может быть либо sentence, либо segment.
        """
        self.asr = asr
        self.tokenize = asr.tokenizer
        self.logfile = logfile
        self.confidence_validation = asr.confidence_validation
        self.global_time_offset = 0.0
        self.init()

        self.buffer_trimming_way = asr.buffer_trimming
        self.buffer_trimming_sec = asr.buffer_trimming_sec

        if self.buffer_trimming_way not in ["sentence", "segment"]:
            raise ValueError("buffer_trimming должен быть либо 'sentence', либо 'segment'")
        if self.buffer_trimming_sec <= 0:
            raise ValueError("buffer_trimming_sec должен быть положительным")
        elif self.buffer_trimming_sec > 30:
            logger.warning(
                f"buffer_trimming_sec установлен на {self.buffer_trimming_sec}, Это очень долго. Это может вызвать ООМ."
            )

    def init(self, offset: Optional[float] = None):
        """Инициализирует или сбросывает буферы обработки."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile, confidence_validation=self.confidence_validation)
        self.buffer_time_offset = offset if offset is not None else 0.0
        self.transcript_buffer.last_committed_time = self.buffer_time_offset
        self.committed: List[ASRToken] = []
        self.time_of_last_asr_output = 0.0

    def get_audio_buffer_end_time(self) -> float:
        """Возвращает абсолютное время окончания текущего audio_buffer."""
        return self.buffer_time_offset + (len(self.audio_buffer) / self.SAMPLING_RATE)

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: Optional[float] = None):
        """Добавить аудио чанк (массив numpy) к текущему аудиобуферу."""
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def insert_silence(self, silence_duration, offset):
        """
        Если пауза длится более 3 секунд, мы полностью очищаем контекст. В противном случае мы просто вставляем небольшую паузу и сдвигаем last_attend_frame.
        """
        # if self.transcript_buffer.buffer:
        #     self.committed.extend(self.transcript_buffer.buffer)
        #     self.transcript_buffer.buffer = []
            
        if True: #silence_duration < 3: #we want the last audio to be treated to not have a gap. could also be handled in the future in ends_with_silence.
            gap_silence = np.zeros(int(16000 * silence_duration), dtype=np.int16)
            self.insert_audio_chunk(gap_silence)
        else:
            self.init(offset=silence_duration + offset)
        self.global_time_offset += silence_duration

    def prompt(self) -> Tuple[str, str]:
        """
        Возвращает кортеж: (prompt, context), где:
        - prompt — это 200-символьный суффикс закомиченного текста, который находится за пределами текущего аудиобуфера.
        - context — это зафиксированный текст в текущем аудиобуфере.
        """
        k = len(self.committed)
        while k > 0 and self.committed[k - 1].end > self.buffer_time_offset:
            k -= 1

        prompt_tokens = self.committed[:k]
        prompt_words = [token.text for token in prompt_tokens]
        prompt_list = []
        length_count = 0
        # Используйте последние слова, пока не достигнете 200 символов.
        while prompt_words and length_count < 200:
            word = prompt_words.pop(-1)
            length_count += len(word) + 1
            prompt_list.append(word)
        non_prompt_tokens = self.committed[k:]
        context_text = self.asr.sep.join(token.text for token in non_prompt_tokens)
        return self.asr.sep.join(prompt_list[::-1]), context_text

    def get_buffer(self):
        """Получить непроверенный буфер в строковом формате."""
        return self.concatenate_tokens(self.transcript_buffer.buffer)
        

    def process_iter(self, language: str = None) -> Tuple[List[ASRToken], float]:
        """Обрабатывает текущий аудиобуфер.
        Возвращает кортеж: (список закомиченных ASRToken, float представляющее обработанный на данный момент аудио).
        """
        current_audio_processed_upto = self.get_audio_buffer_end_time()
        prompt_text, _ = self.prompt()
        logger.debug(
            f"Transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:.2f} seconds from {self.buffer_time_offset:.2f}"
        )
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt_text, language=language)
        #logger.info(f"{res=}")
        tokens = self.asr.ts_words(res)
        self.transcript_buffer.insert(tokens, self.buffer_time_offset)
        committed_tokens = self.transcript_buffer.flush()
        self.committed.extend(committed_tokens)

        if committed_tokens:
            self.time_of_last_asr_output = self.committed[-1].end

        completed = self.concatenate_tokens(committed_tokens)
        # Текст подтвержден - это тот текст, который он окончательно "зафиксировал" и который уже не изменится.
        logger.debug(f"Текст подтвержден: {completed.text}")
        incomp = self.concatenate_tokens(self.transcript_buffer.buffer)
        # Неподтвержденный - это текст, который он только начал печатать. Он может его исправить, удалить или переписать в следующую секунду. Это "черновик" или "хвост" гипотезы.
        logger.debug(f"Неподтвержденный: {incomp.text}")

        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE
        if not committed_tokens and buffer_duration > self.buffer_trimming_sec:
            time_since_last_output = self.get_audio_buffer_end_time() - self.time_of_last_asr_output
            if time_since_last_output > self.buffer_trimming_sec:
                """Сообщение появляется, когда ASR долго не выдаёт подтверждённые токены: если время с момента последнего “закомиченного” вывода
                превышает порог buffer_trimming_sec, процессор логирует предупреждение и сбрасывает буфер,
                чтобы не зависнуть на чрезмерно длинной обработке."""
                logger.warning(
                    f"ASR не выдал подтвержденных токенов за {time_since_last_output:.2f}s. "
                    f"Сброс буфера для предотвращения зависания из-за слишком долгой обработки."
                )
                self.init(offset=self.get_audio_buffer_end_time())
                return [], current_audio_processed_upto

        if committed_tokens and self.buffer_trimming_way == "sentence":
            if len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec:
                self.chunk_completed_sentence()

        s = self.buffer_trimming_sec if self.buffer_trimming_way == "segment" else 30
        if len(self.audio_buffer) / self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)
            logger.debug("Chunking segment")
        logger.debug(
            f"Длина аудиобуфера сейчас: {len(self.audio_buffer)/self.SAMPLING_RATE:.2f} секунды"
        )
        if self.global_time_offset:
            for token in committed_tokens:
                token = token.with_offset(self.global_time_offset)
        return committed_tokens, current_audio_processed_upto

    def chunk_completed_sentence(self):
        """
        Если закомиченные токены образуют как минимум два предложения, аудиобуфер делится на чанки в момент окончания предпоследнего предложения.
        Также гарантируется деление на чанки, если аудиобуфер превышает ограничение по времени.
        """
        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE        
        if not self.committed:
            if buffer_duration > self.buffer_trimming_sec:
                chunk_time = self.buffer_time_offset + (buffer_duration / 2)
                logger.debug(f"--- Речь не обнаружена, принудительное разделение на фрагменты {chunk_time:.2f}")
                self.chunk_at(chunk_time)
            return
        
        logger.debug("ЗАВЕРШЕННОЕ ПРЕДЛОЖЕНИЕ: " + " ".join(token.text for token in self.committed))
        sentences = self.words_to_sentences(self.committed)
        for sentence in sentences:
            logger.debug(f"\tSentence: {sentence.text}")
        
        chunk_done = False
        if len(sentences) >= 2:
            while len(sentences) > 2:
                sentences.pop(0)
            chunk_time = sentences[-2].end
            logger.debug(f"--- Предложение разбито на чанки {chunk_time:.2f}")
            self.chunk_at(chunk_time)
            chunk_done = True
        
        if not chunk_done and buffer_duration > self.buffer_trimming_sec:
            last_committed_time = self.committed[-1].end
            logger.debug(f"--- Недостаточно предложений, разделение на чанки в последний выделенный момент {last_committed_time:.2f}")
            self.chunk_at(last_committed_time)

    def chunk_completed_segment(self, res):
        """
        Разделение аудиобуфера на segment-end на основе временных меток конца сегмента, полученных от ASR.
        Также гарантирует разделение на чанки, если аудиобуфер превышает ограничение по времени.
        """
        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE        
        if not self.committed:
            if buffer_duration > self.buffer_trimming_sec:
                chunk_time = self.buffer_time_offset + (buffer_duration / 2)
                logger.debug(f"--- Речь не обнаружена, принудительное разделение на фрагменты {chunk_time:.2f}")
                self.chunk_at(chunk_time)
            return
        
        logger.debug("Обработка выделенных токенов для сегментации")
        ends = self.asr.segments_end_ts(res)
        last_committed_time = self.committed[-1].end        
        chunk_done = False
        if len(ends) > 1:
            logger.debug("Доступно несколько сегментов для разделения на чанки")
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > last_committed_time:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= last_committed_time:
                logger.debug(f"--- Сегмент разделен на чанки {e:.2f}")
                self.chunk_at(e)
                chunk_done = True
            else:
                logger.debug("--- Последний сегмент не в пределах выделенной области")
        else:
            logger.debug("--- Недостаточно сегментов для разбиения на чанки")
        
        if not chunk_done and buffer_duration > self.buffer_trimming_sec:
            logger.debug(f"--- Буфер слишком большой, фрагментация происходит в последний зафиксированный момент времени {last_committed_time:.2f}")
            self.chunk_at(last_committed_time)
        
        logger.debug("Разделение сегмента завершено")
        
    def chunk_at(self, time: float):
        """Обрежет как гипотезу, так и аудиобуфер в заданное время."""
        logger.debug(f"Разделение на чанки {time:.2f}s")
        logger.debug(
            f"Длина аудиобуфера перед разделением на фрагменты: {len(self.audio_buffer)/self.SAMPLING_RATE:.2f}s"
        )
        self.transcript_buffer.pop_committed(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.SAMPLING_RATE):]
        self.buffer_time_offset = time
        logger.debug(
            f"Длина аудиобуфера после разбиения на фрагменты: {len(self.audio_buffer)/self.SAMPLING_RATE:.2f}s"
        )

    def words_to_sentences(self, tokens: List[ASRToken]) -> List[Sentence]:
        """
        Преобразует список токенов в список объектов Sentence, используя предоставленный токенизатор предложений.
        """
        if not tokens:
            return []

        full_text = " ".join(token.text for token in tokens)

        if self.tokenize:
            try:
                sentence_texts = self.tokenize(full_text)
            except Exception as e:
                # Некоторые токенизаторы (например, MosesSentenceSplitter) ожидают ввода списка.
                try:
                    sentence_texts = self.tokenize([full_text])
                except Exception as e2:
                    raise ValueError("Tokenization failed") from e2
        else:
            sentence_texts = [full_text]

        sentences: List[Sentence] = []
        token_index = 0
        for sent_text in sentence_texts:
            sent_text = sent_text.strip()
            if not sent_text:
                continue
            sent_tokens = []
            accumulated = ""
            # Накапливайте токены до тех пор, пока их длина не будет примерно соответствовать длине текста предложения.
            while token_index < len(tokens) and len(accumulated) < len(sent_text):
                token = tokens[token_index]
                accumulated = (accumulated + " " + token.text).strip() if accumulated else token.text
                sent_tokens.append(token)
                token_index += 1
            if sent_tokens:
                sentence = Sentence(
                    start=sent_tokens[0].start,
                    end=sent_tokens[-1].end,
                    text=" ".join(t.text for t in sent_tokens),
                )
                sentences.append(sentence)
        return sentences
    
    def finish(self) -> Tuple[List[ASRToken], float]:
        """
        Сбросить оставшейся transcript после завершения обработки.
        Возвращает кортеж: (список оставшихся объектов ASRToken, float представляющее собой последний обработанный аудиофайл).
        """
        remaining_tokens = self.transcript_buffer.buffer
        logger.debug(f"Final non-committed tokens: {remaining_tokens}")
        final_processed_upto = self.buffer_time_offset + (len(self.audio_buffer) / self.SAMPLING_RATE)
        self.buffer_time_offset = final_processed_upto
        return remaining_tokens, final_processed_upto

    def concatenate_tokens(
        self,
        tokens: List[ASRToken],
        sep: Optional[str] = None,
        offset: float = 0
    ) -> Transcript:
        sep = sep if sep is not None else self.asr.sep
        text = sep.join(token.text for token in tokens)
        probability = sum(token.probability for token in tokens if token.probability) / len(tokens) if tokens else None
        if tokens:
            start = offset + tokens[0].start
            end = offset + tokens[-1].end
        else:
            start = None
            end = None
        return Transcript(start, end, text, probability=probability)
