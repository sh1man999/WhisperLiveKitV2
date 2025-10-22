import asyncio
import os

import websockets
import numpy as np
import librosa
import sounddevice as sd
import json
import time
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)


async def send_audio(websocket, source, chunk_size_s=1.0, sample_rate=16000):
    """Send audio chunks (file or mic) to websocket"""
    if source == "mic":
        logger.info("Streaming from microphone...")
        q = asyncio.Queue()

        def callback(indata, frames, time_info, status):
            q.put_nowait(indata.copy())

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                await _send_chunk(websocket, chunk, sample_rate)
                await asyncio.sleep(chunk_size_s)

    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        audio, sample_rate = librosa.load(path, sr=16000, mono=True)
        duration = len(audio) / sample_rate
        chunk_size = int(sample_rate * chunk_size_s)
        logger.info(f"Streaming {path.name} ({duration:.2f}s, sample_rate={sample_rate}, chunk={chunk_size_s:.2f}s)")

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) == 0:
                continue
            await _send_chunk(websocket, chunk, sample_rate)
        await asyncio.sleep(100)
        await websocket.send(b"")  # End of stream
        return duration


async def _send_chunk(websocket, chunk, sr_in):
    """Encode PCM chunk to bytes and send s16le"""
    chunk_int16 = (chunk * 32768).astype(np.int16)
    await websocket.send(chunk_int16.tobytes())


async def receive_updates(websocket, first_token_event):  # noqa: C901, ANN001
    """Receive server responses and mark first token."""
    # ANSI color codes
    GRAY = "\033[90m"
    RESET = "\033[0m"

    displayed_lines = {}  # Map line_id -> line_text
    texts = {}
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

                text = line['text'].strip()
                line_display = f"ID:{line['id']} {line['start']} - {line['end']} Speaker {line['speaker']}: {text}"
                texts[line_id] = text

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
                for text in texts.values():
                    print(text, end=" ")
                break

        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Connection closed normally")
            break
        except Exception as e:
            logger.error(f"Error receiving updates: {e}")
            break


async def test_server(source, host="localhost", port=8000, chunk_size=1.0):
    """Main pipeline: send + receive + metrics"""
    uri = f"ws://{host}:{port}/asr?language=ru"
    async with websockets.connect(uri) as ws:
        logger.info(f"Connected to {uri}")
        first_token_event = asyncio.Event()
        recv_task = asyncio.create_task(receive_updates(ws, first_token_event))

        start_time = time.time()
        duration = await send_audio(ws, source, chunk_size_s=chunk_size)

        try:
            await asyncio.wait_for(first_token_event.wait(), timeout=30)
            first_latency = time.time() - start_time
        except asyncio.TimeoutError:
            first_latency = None

        await recv_task
        total_time = time.time() - start_time
        rtf = total_time / duration if duration else None

        print("\n========== METRICS ==========")
        print(f"First Token Latency: {first_latency:.3f}s" if first_latency else "No token received")
        print(f"Total Time: {total_time:.3f}s")
        print(f"Real Time Factor: {rtf:.3f}" if rtf else "RTF: undefined (mic input)")
        print("=============================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASR Streaming Client, you should start the Server \
            with pcm-input: whisperlivekit-server  --pcm-input")
    parser.add_argument("--source", type=str, default=os.path.join(Path(__file__).parent, "assets", "test.flac"),
                        help="Audio file path or 'mic' for microphone")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunk_size", type=float, default=1.0,
                        help="Chunk size in seconds (default: 1.0)")
    args = parser.parse_args()

    asyncio.run(test_server(
        source=args.source,
        host=args.host,
        port=args.port,
        chunk_size=args.chunk_size,
    ))
