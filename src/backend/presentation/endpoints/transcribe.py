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
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response.to_dict())
        # when the results_generator finishes it means all audio has been processed
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results (client likely closed connection).")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")

@transcribe_router.websocket("/asr")
@inject
async def transcribe(websocket: WebSocket, transcription_engine: FromDishka[TranscriptionEngine]):
    await websocket.accept()
    pcm_input = True # декодирование аудио на стороне клиента
    logger.info("WebSocket connection opened.")
    language = websocket.query_params.get("language", "auto")
    url = websocket.query_params.get("url")
    try:
        await websocket.send_json({"type": "config", "useAudioWorklet": pcm_input})
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
        language=language,
        url=url,
        pcm_input=pcm_input
    )
    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(
        handle_websocket_results(websocket, results_generator)
    )

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if "bytes" in str(e):
            logger.warning("Client has closed the connection.")
        else:
            logger.error(
                f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True
            )
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(
            f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True
        )
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")

        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")
