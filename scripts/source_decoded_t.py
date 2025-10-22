#!/usr/bin/env python3
"""
Test script for streaming audio from URL to ASR backend.

This script fetches audio from a URL stream locally using FFmpeg,
then sends PCM chunks to the ASR WebSocket endpoint in batches,
simulating a real client streaming scenario.

Usage:
    python source_decoded_t.py --url <stream_url>
    python source_decoded_t.py --url <stream_url> --chunk_size 1.0
    python source_decoded_t.py --file <local_audio_file>
"""

import asyncio
import logging
import argparse
import json
import time
import subprocess
from pathlib import Path
import numpy as np

import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AudioStreamReader:
    """Reads audio from URL/file using FFmpeg and provides PCM chunks."""

    def __init__(self, source, sample_rate=16000, channels=1):
        self.source = source
        self.sample_rate = sample_rate
        self.channels = channels
        self.process = None

    async def start(self):
        """Start FFmpeg process to read audio."""
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', self.source,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(self.sample_rate),
            '-ac', str(self.channels),
            '-loglevel', 'error',
            'pipe:1'
        ]

        logger.info(f"Starting FFmpeg to read from: {self.source}")
        logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

        try:
            self.process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            logger.info("FFmpeg process started")
            return True
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            return False

    async def read_chunk(self, chunk_size_bytes):
        """Read a chunk of PCM data from FFmpeg stdout."""
        if not self.process or not self.process.stdout:
            return None

        try:
            chunk = await self.process.stdout.read(chunk_size_bytes)
            return chunk if chunk else None
        except Exception as e:
            logger.error(f"Error reading from FFmpeg: {e}")
            return None

    async def stop(self):
        """Stop FFmpeg process."""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
                logger.info("FFmpeg process stopped")
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
                logger.warning("FFmpeg process killed (timeout)")
            except Exception as e:
                logger.error(f"Error stopping FFmpeg: {e}")


async def send_audio_stream(websocket, stream_reader, chunk_size_s=1.0, sample_rate=16000):
    """
    Read audio from stream and send chunks to WebSocket.

    Args:
        websocket: WebSocket connection
        stream_reader: AudioStreamReader instance
        chunk_size_s: Chunk duration in seconds
        sample_rate: Audio sample rate
    """
    bytes_per_sample = 2  # int16
    chunk_size_bytes = int(sample_rate * chunk_size_s * bytes_per_sample)

    logger.info(f"Starting to send audio chunks (chunk_size={chunk_size_s}s, {chunk_size_bytes} bytes)")

    total_bytes = 0
    chunk_count = 0
    start_time = time.time()

    try:
        while True:
            chunk = await stream_reader.read_chunk(chunk_size_bytes)

            if not chunk:
                logger.info("No more data from stream")
                break

            # Send raw PCM bytes to server
            await websocket.send(chunk)

            total_bytes += len(chunk)
            chunk_count += 1

            if chunk_count % 10 == 0:
                duration = total_bytes / (sample_rate * bytes_per_sample)
                elapsed = time.time() - start_time
                # logger.info(
                #     f"Sent {chunk_count} chunks, {duration:.1f}s of audio "
                #     f"in {elapsed:.1f}s elapsed"
                # )

            # Small delay to avoid overwhelming the server
            await asyncio.sleep(0.01)

        # Send empty message to signal end of stream
        await websocket.send(b"")
        logger.info(f"Stream complete: {chunk_count} chunks, {total_bytes} bytes sent")

    except Exception as e:
        logger.error(f"Error sending audio: {e}", exc_info=True)
        raise


def parse_time_to_seconds(time_str):
    """Parse time string in format '0:01:23' or float to seconds."""
    if isinstance(time_str, (int, float)):
        return float(time_str)

    # Parse format like '0:01:23' (hours:minutes:seconds)
    parts = str(time_str).split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    else:
        return float(time_str)


async def receive_updates(websocket, first_token_event, start_time):
    """Receive server responses and mark first token."""
    # ANSI color codes
    GRAY = "\033[90m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    displayed_lines = {}  # Map line_id -> (text, start, end, speaker) to detect content changes
    last_line_id = None  # Track the last finalized line ID
    last_buffer = ""  # Track last buffer to avoid redundant updates

    while True:
        try:
            msg = await websocket.recv()
            resp = json.loads(msg)

            # Check for errors
            if resp.get("status") == "error":
                error_msg = resp.get("error", "Unknown error")
                print(f"\n{YELLOW}Server error: {error_msg}{RESET}")
                logger.error(f"Server error: {error_msg}")
                break

            # Process finalized lines
            lines = resp.get("lines", [])

            # Show ALL lines from server (no filtering)
            valid_lines = [
                line
                for line in lines
                if line.get("speaker", -1) >= 0  # Valid speaker ID
                and line.get("speaker", -1) != -2  # Not a dummy line
                and line.get("id") is not None  # Has valid ID
            ]

            current_seen_lines = set()

            for line in valid_lines:
                line_id = line.get("id")
                if line_id is None:
                    continue

                current_seen_lines.add(line_id)

                # Parse time (handles both float and '0:01:23' format)
                start = parse_time_to_seconds(line['start'])
                end = parse_time_to_seconds(line['end'])
                text = line['text'].strip()
                speaker = line['speaker']

                # Create a key to check if content changed (not metrics)
                content_key = (text, start, end, speaker)

                # Check if this line is now finalized (no longer in last update or content stopped changing)
                content_changed = displayed_lines.get(line_id) != content_key

                if line_id not in displayed_lines:
                    # New line - print without metrics
                    text_line = text if speaker != -2 else "Тишина"
                    line_display = (
                        f"{GREEN}ID:{line_id}{RESET} "
                        f"{start:.2f}s - {end:.2f}s "
                        f"Speaker {speaker}: {text_line}"
                    )
                    print(f"\n{line_display}", end="", flush=True)
                    displayed_lines[line_id] = content_key
                    last_line_id = line_id
                    if not first_token_event.is_set():
                        first_token_event.set()

                elif content_changed:
                    # Line content changed - update without metrics
                    text_line = text if speaker != -2 else "Тишина"
                    line_display = (
                        f"{GREEN}ID:{line_id}{RESET} "
                        f"{start:.2f}s - {end:.2f}s "
                        f"Speaker {speaker}: {text_line}"
                    )
                    if line_id == last_line_id:
                        print(f"\r{line_display}", end="", flush=True)
                    else:
                        print(f"\n{line_display}", end="", flush=True)
                        last_line_id = line_id
                    displayed_lines[line_id] = content_key


            # Display buffer on the same line as the last finalized line, in gray
            buffer_trans = resp.get("buffer_transcription", "")

            if last_line_id is not None:
                # Get the last line text
                content_key = displayed_lines.get(last_line_id)
                if content_key:
                    text, start, end, speaker = content_key
                    text_line = text if speaker != -2 else "Тишина"
                    last_line_text = (
                        f"{GREEN}ID:{last_line_id}{RESET} "
                        f"{start:.2f}s - {end:.2f}s "
                        f"Speaker {speaker}: {text_line}"
                    )

                    # Update display with buffer appended in gray (without metrics)
                    if buffer_trans:
                        combined = f"\r{last_line_text} {GRAY}{buffer_trans}{RESET}"
                        print(combined, end="", flush=True)
                        last_buffer = buffer_trans
                    elif last_buffer:
                        # Buffer was cleared, redraw just the line
                        print(f"\r{last_line_text}", end="", flush=True)
                        last_buffer = ""

            if resp.get("type") == "ready_to_stop":
                print("\n")
                logger.info("Stream processing complete")
                break

        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Connection closed normally")
            break
        except Exception as e:
            logger.error(f"Error receiving updates: {e}")
            break


async def test_url_stream(source, host="localhost", port=8000, chunk_size=1.0, language="ru"):
    """
    Main test pipeline: read from URL stream, send to server, receive results.

    Args:
        source: URL or file path to audio source
        host: ASR server host
        port: ASR server port
        chunk_size: Audio chunk size in seconds
        language: Language code
    """
    uri = f"ws://{host}:{port}/asr?language={language}"

    # Create stream reader
    stream_reader = AudioStreamReader(source, sample_rate=16000, channels=1)

    if not await stream_reader.start():
        logger.error("Failed to start audio stream reader")
        return

    try:
        async with websockets.connect(uri) as ws:
            logger.info(f"Connected to {uri}")

            first_token_event = asyncio.Event()
            start_time = time.time()

            # Start receiving results
            recv_task = asyncio.create_task(
                receive_updates(ws, first_token_event, start_time)
            )

            # Start sending audio
            send_task = asyncio.create_task(
                send_audio_stream(ws, stream_reader, chunk_size_s=chunk_size)
            )

            # Wait for first token
            try:
                await asyncio.wait_for(first_token_event.wait(), timeout=30)
            except asyncio.TimeoutError:
                logger.warning("No transcription received within 30 seconds")

            # Wait for sending to complete
            await send_task

            # Wait for all results
            await recv_task


    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
    finally:
        await stream_reader.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Test ASR by streaming audio from URL/file to server. "
                    "Audio is fetched locally via FFmpeg and sent as PCM chunks."
    )

    # Source options
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--url",
        type=str,
        help="Stream URL (http/https/rtmp/hls/etc supported by ffmpeg)"
    )
    source_group.add_argument(
        "--file",
        type=str,
        help="Local audio file path"
    )

    # Server options
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="ASR server host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="ASR server port (default: 8000)"
    )
    parser.add_argument(
        "--chunk_size",
        type=float,
        default=1.0,
        help="Chunk size in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ru",
        help="Language code (default: ru)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (show all INFO messages)"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.INFO)

    # Determine source
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {args.file}")
            return
        source = str(file_path.absolute())
    elif args.url:
        source = args.url
    else:
        # Default test stream
        source = "https://vgtrkregion-reg.cdnvideo.ru/vgtrk/0/russia24-sd/index.m3u8"
        logger.info("No source specified, using default test stream")

    try:
        asyncio.run(test_url_stream(
            source=source,
            host=args.host,
            port=args.port,
            chunk_size=args.chunk_size,
            language=args.language,
        ))
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")


if __name__ == "__main__":
    main()
