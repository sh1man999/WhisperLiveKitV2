import logging
import re
from typing import Final

from whisperlivekit.remove_silences import handle_silences
from whisperlivekit.timed_objects import Line, format_time

logger = logging.getLogger(__name__)
# Уникальный маркер для временной замены точек в исключениях (сокращениях)
_TEMP_DOT_MARKER: Final[str] = "_TEMP_DOT_MARKER_"
# Словарь для замены точек в сокращениях на временный маркер.
_ABBREVIATION_REPLACE_DICT: Final[dict[str, str]] = {
    "и т.д.": f"и т{_TEMP_DOT_MARKER}д",
    "и т.п.": f"и т{_TEMP_DOT_MARKER}п",
    "т.е.": f"т{_TEMP_DOT_MARKER}е",
    "т.н.": f"т{_TEMP_DOT_MARKER}н",
    "Т.е.": f"Т{_TEMP_DOT_MARKER}е",
    "Т.н.": f"Т{_TEMP_DOT_MARKER}н",
    "и т. д.": f"и т{_TEMP_DOT_MARKER} д",
    "и т. п.": f"и т{_TEMP_DOT_MARKER} п",
    "т. е.": f"т{_TEMP_DOT_MARKER} е",
    "т. н.": f"т{_TEMP_DOT_MARKER} н",
    "Т. е.": f"Т{_TEMP_DOT_MARKER} е",
    "Т. н.": f"Т{_TEMP_DOT_MARKER} н",
}
# Словарь для восстановления оригинальных точек в сокращениях.
_ABBREVIATION_RESTORE_DICT: Final[dict[str, str]] = {
    value: key for key, value in _ABBREVIATION_REPLACE_DICT.items()
}
# Скомпилированное регулярное выражение для поиска букв, которые нужно сделать заглавными.
_CAPITALIZE_PATTERN = re.compile(r"([.!?]\s*|[.!?]\n\[.*?\]\s*)([a-zа-я])")
_CHECK_AROUND = 4
_DEBUG = False

def capitalize_after_delimiters(text: str) -> str:
    if not text:  # Обработка пустой строки
        return ""

    # 1. Защита точек в известных сокращениях путем замены на временный маркер
    for original_abbr, marked_abbr in _ABBREVIATION_REPLACE_DICT.items():
        text = text.replace(original_abbr, marked_abbr)

    # 2. Применение капитализации после разделителей предложений
    # _CAPITALIZE_PATTERN находит фразу-разделитель (группа 1)
    # и следующую за ней строчную букву (группа 2).
    # лямбда затем заменяет это совпадение, делая букву заглавной.
    text = _CAPITALIZE_PATTERN.sub(lambda m: m.group(1) + m.group(2).upper(), text)

    # 3. Восстановление оригинальных точек в сокращениях
    for marked_abbr, original_abbr in _ABBREVIATION_RESTORE_DICT.items():
        text = text.replace(marked_abbr, original_abbr)

    return text


def is_punctuation(token):
    if token.is_punctuation():
        return True
    return False

def next_punctuation_change(i, tokens):
    for ind in range(i+1, min(len(tokens), i + _CHECK_AROUND + 1)):
        if is_punctuation(tokens[ind]):
            return ind
    return None

def next_speaker_change(i, tokens, speaker):
    for ind in range(i-1, max(0, i - _CHECK_AROUND) - 1, -1):
        token = tokens[ind]
        if is_punctuation(token):
            break
        if token.speaker != speaker:
            return ind, token.speaker
    return None, speaker

def new_line(
        token,
):
    return Line(
        speaker = token.corrected_speaker,
        text = token.text + (f"[{format_time(token.start)} : {format_time(token.end)}]" if _DEBUG else ""),
        start = token.start,
        end = token.end,
        detected_language=token.detected_language
    )

def append_token_to_last_line(lines, sep, token):
    if not lines:
        lines.append(new_line(token))
    else:
        if token.text:
            lines[-1].text += sep + token.text + (f"[{format_time(token.start)} : {format_time(token.end)}]" if _DEBUG else "")
            lines[-1].end = token.end
        if not lines[-1].detected_language and token.detected_language:
            lines[-1].detected_language = token.detected_language


def format_output(state, silence, current_time, args, sep):
    diarization = args.diarization
    disable_punctuation_split = args.disable_punctuation_split
    tokens = state.tokens
    last_validated_token = state.last_validated_token

    previous_speaker = 1
    undiarized_text = []
    tokens = handle_silences(tokens, current_time, silence)
    last_punctuation = None
    for i, token in enumerate(tokens[last_validated_token:]):
        speaker = int(token.speaker)
        token.corrected_speaker = speaker
        if not diarization:
            if speaker == -1: #Speaker -1 means no attributed by diarization. In the frontend, it should appear under 'Speaker 1'
                token.corrected_speaker = 1
                token.validated_speaker = True
        else:
            # if token.end > end_attributed_speaker and token.speaker != -2:
            #     if tokens[-1].speaker == -2:  #if it finishes by a silence, we want to append the undiarized text to the last speaker.
            #         token.corrected_speaker = previous_speaker
            #     else:
            #         undiarized_text.append(token.text)
            #         continue
            # else:
            if is_punctuation(token):
                last_punctuation = i

            if last_punctuation == i-1:
                if token.speaker != previous_speaker:
                    token.validated_speaker = True
                    # perfect, diarization perfectly aligned
                    last_punctuation = None
                else:
                    speaker_change_pos, new_speaker = next_speaker_change(i, tokens, speaker)
                    if speaker_change_pos:
                        # Corrects delay:
                        # That was the idea. <Okay> haha |SPLIT SPEAKER| that's a good one
                        # should become:
                        # That was the idea. |SPLIT SPEAKER| <Okay> haha that's a good one
                        token.corrected_speaker = new_speaker
                        token.validated_speaker = True
            elif speaker != previous_speaker:
                if not (speaker == -2 or previous_speaker == -2):
                    if next_punctuation_change(i, tokens):
                        # Corrects advance:
                        # Are you |SPLIT SPEAKER| <okay>? yeah, sure. Absolutely
                        # should become:
                        # Are you <okay>? |SPLIT SPEAKER| yeah, sure. Absolutely
                        token.corrected_speaker = previous_speaker
                        token.validated_speaker = True
                    else: #Problematic, except if the language has no punctuation. We append to previous line, except if disable_punctuation_split is set to True.
                        if not disable_punctuation_split:
                            token.corrected_speaker = previous_speaker
                            token.validated_speaker = False
        if token.validated_speaker:
            state.last_validated_token = i
        previous_speaker = token.corrected_speaker

    previous_speaker = 1

    lines = []
    for token in tokens:
        split_now = False
        if getattr(args, 'split_on_punctuation_for_display', False) and lines:
            line_duration = lines[-1].end - lines[-1].start
            if line_duration > 5:
                last_line_text = lines[-1].text.strip()
                if last_line_text:
                    if last_line_text.endswith((".", "?", "!")):
                        split_now = True
                    # words_in_line = len(lines[-1].text.split())
                    # if words_in_line >= 15:
                    #     split_now = True

        if split_now or (lines and int(token.corrected_speaker) != int(previous_speaker)):
            lines.append(new_line(token))
        else:
            append_token_to_last_line(lines, sep, token)

        previous_speaker = token.corrected_speaker


    for i, line in enumerate(lines):
        should_capitalize = False
        if len(lines) == 1:
            should_capitalize = True
        else:
            prev_text = lines[i-1].text.rstrip()
            if prev_text and prev_text[-1] in (".", "?", "!"):
                should_capitalize = True
        if should_capitalize:
            # Capitalize the very first letter
            stripped_text = line.text.lstrip()
            if stripped_text:
                capitalized_text = stripped_text[0].upper() + stripped_text[1:]
                line.text = line.text[:len(line.text) - len(stripped_text)] + capitalized_text

        # Then apply the user's logic for subsequent sentences
        line.text = capitalize_after_delimiters(line.text)

    if state.buffer_transcription and lines:
        lines[-1].end = max(state.buffer_transcription.end, lines[-1].end)

    return lines, undiarized_text