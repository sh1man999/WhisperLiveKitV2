import asyncio
import logging

from dishka import FromDishka
from dishka.integrations.fastapi import inject
from fastapi import APIRouter
from starlette.websockets import WebSocket, WebSocketDisconnect
from whisperlivekit import AudioProcessor, TranscriptionEngine

transcribe_router = APIRouter()

logger = logging.getLogger(__name__)

async def handle_websocket_results(websocket, results_generator):
    """Получает результаты от аудиопроцессора и отправляет их через WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response.to_dict())
        logger.info("Генератор результатов завершён. Отправляю клиенту сообщение «ready_to_stop».")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info("WebSocket отключился во время обработки результатов (вероятно, клиент закрыл соединение).")
    except Exception as e:
        logger.exception(f"Ошибка в обработчике результатов WebSocket: {e}")

@transcribe_router.websocket("/asr")
@inject
async def transcribe(websocket: WebSocket, transcription_engine: FromDishka[TranscriptionEngine]):
    await websocket.accept()
    logger.info("WebSocket-соединение открыто.")
    language = websocket.query_params.get("language", "auto")
    url = websocket.query_params.get("url")
    try:
        await websocket.send_json({"type": "config"})
    except Exception as e:
        logger.warning(f"Не удалось отправить конфигурацию клиенту: {e}")
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
        language=language,
        url=url
    )
    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(
        handle_websocket_results(websocket, results_generator)
    )

    try:
        if url:
            await websocket_task
        else:
            # Для live audio input, продолжайте получать байты от клиента.
            while True:
                message = await websocket.receive_bytes()
                await audio_processor.process_audio(message)
    except KeyError as e:
        if "bytes" in str(e):
            logger.warning("Клиент закрыл соединение.")
        else:
            logger.error(
                f"Неожиданная ошибка KeyError в websocket_endpoint: {e}", exc_info=True
            )
    except WebSocketDisconnect:
        logger.info("WebSocket был отключен клиентом во время цикла получения сообщений.")
    except Exception as e:
        logger.error(
            f"Неожиданная ошибка в основном цикле websocket_endpoint: {e}", exc_info=True
        )
    finally:
        logger.info("Очистка конечной точки WebSocket...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("Задача обработчика результатов WebSocket была отменена.")
        except Exception as e:
            logger.warning(f"Исключение при ожидании завершения websocket_task: {e}")

        await audio_processor.cleanup()
        logger.info("Конечная точка WebSocket успешно очищена.")
