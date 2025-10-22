import asyncio
import logging
import argparse
import json
import time
import urllib.parse

import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)


async def receive_updates(websocket, first_token_event):  # noqa: C901, ANN001
    """Receive server responses and mark first token."""
    # ANSI color codes
    GRAY = "\033[90m"
    RESET = "\033[0m"

    displayed_lines = {}  # Map line_id -> line_text
    last_line_id = None  # Track the last finalized line ID
    last_buffer = ""  # Track last buffer to avoid redundant updates

    while True:
        try:
            msg = await websocket.recv()
            resp = json.loads(msg)

            # Process finalized lines
            lines = resp.get("lines", [])
            valid_lines = [
                line for line in lines
                if line.get("text", "").strip() and line.get("speaker", -1) >= 0
            ]

            for line in valid_lines:
                line_id = line.get("id")
                if line_id is None:
                    continue

                line_display = f"[{line_id}] [{line['start']} - {line['end']}] [Speaker {line['speaker']}]: {line['text'].strip()}"

                if line_id not in displayed_lines:
                    # New line - always print with newline
                    print(f"\n{line_display}", end="", flush=True)  # noqa: T201
                    displayed_lines[line_id] = line_display
                    last_line_id = line_id
                    if not first_token_event.is_set():
                        first_token_event.set()
                elif displayed_lines[line_id] != line_display:
                    # Line updated - use \r only if THIS line is the last one we displayed
                    if line_id == last_line_id:
                        print(f"\r{line_display}", end="", flush=True)  # noqa: T201
                    else:
                        print(f"\n{line_display}", end="", flush=True)  # noqa: T201
                        last_line_id = line_id
                    displayed_lines[line_id] = line_display

            # Display buffer on the same line as the last finalized line, in gray
            buffer_trans = resp.get("buffer_transcription", "")

            if last_line_id is not None:
                # Get the last line text
                last_line_text = displayed_lines.get(last_line_id, "")

                # Update display with buffer appended in gray
                if buffer_trans:
                    combined = f"\r{last_line_text} {GRAY}{buffer_trans}{RESET}"
                    print(combined, end="", flush=True)  # noqa: T201
                    last_buffer = buffer_trans
                elif last_buffer:
                    # Buffer was cleared, redraw just the line
                    print(f"\r{last_line_text}", end="", flush=True)  # noqa: T201
                    last_buffer = ""

            if resp.get("type") == "ready_to_stop":
                print("\n")  # noqa: T201
                break

        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Connection closed normally")
            break
        except Exception as e:
            logger.error(f"Error receiving updates: {e}")
            break


async def transcribe_stream(stream_url, host="localhost", port=8000, language="ru"):
    """
    Transcribe audio from a stream URL.

    The server will handle downloading and processing the stream.
    We just connect and receive the results.
    """
    # URL-encode the stream URL for query parameter
    encoded_url = urllib.parse.quote(stream_url, safe='')
    uri = f"ws://{host}:{port}/asr?language={language}&url={encoded_url}"

    logger.info(f"Connecting to ASR server...")
    logger.info(f"Stream URL: {stream_url}")

    async with websockets.connect(uri) as ws:
        logger.info(f"Connected to {uri}")
        logger.info("Server is processing the stream...")

        first_token_event = asyncio.Event()
        start_time = time.time()

        # Wait for first token
        recv_task = asyncio.create_task(receive_updates(ws, first_token_event))

        try:
            await asyncio.wait_for(first_token_event.wait(), timeout=60)
            first_latency = time.time() - start_time
        except asyncio.TimeoutError:
            first_latency = None
            logger.warning("No transcription received within 60 seconds")

        # Wait for all results
        await recv_task
        total_time = time.time() - start_time

        print("\n========== METRICS ==========")
        print(f"First Token Latency: {first_latency:.3f}s" if first_latency else "No token received")
        print(f"Total Processing Time: {total_time:.3f}s")
        print("=============================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASR Streaming Client for URL streams. The server handles downloading and processing the stream."
    )
    parser.add_argument("--url", default="https://vgtrkregion-reg.cdnvideo.ru/vgtrk/0/russia24-sd/index.m3u8", type=str,
                        help="Stream URL (http/https/rtmp/etc supported by ffmpeg)")
    parser.add_argument("--host", type=str, default="localhost",
                        help="ASR server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000,
                        help="ASR server port (default: 8000)")
    parser.add_argument("--language", type=str, default="ru",
                        help="Language code (default: ru)")
    args = parser.parse_args()

    asyncio.run(transcribe_stream(
        stream_url=args.url,
        host=args.host,
        port=args.port,
        language=args.language,
    ))
