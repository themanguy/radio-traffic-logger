"""
Stream ingestor: pulls an HTTP audio stream via ffmpeg and yields 16kHz mono
PCM frames as numpy float32 arrays.
"""

import subprocess
import time
import numpy as np
import logging
from collections.abc import Iterator
from config import STREAM_URL, RECONNECT_DELAY_SECONDS

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
# Feed VAD in 512-sample chunks (32 ms) — required by Silero VAD
CHUNK_SAMPLES = 512


def _build_ffmpeg_cmd(url: str) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
        "-i", url,
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-f", "s16le",   # signed 16-bit little-endian PCM
        "pipe:1",
    ]


def stream_pcm(url: str = STREAM_URL) -> Iterator[np.ndarray]:
    """
    Yields numpy float32 arrays of shape (CHUNK_SAMPLES,) at 16kHz.
    Automatically reconnects if the stream drops.
    """
    bytes_per_sample = 2  # s16le
    chunk_bytes = CHUNK_SAMPLES * bytes_per_sample

    while True:
        log.info("Connecting to stream: %s", url)
        try:
            proc = subprocess.Popen(
                _build_ffmpeg_cmd(url),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            buf = b""
            while True:
                data = proc.stdout.read(chunk_bytes * 4)
                if not data:
                    log.warning("Stream ended — reconnecting in %ds", RECONNECT_DELAY_SECONDS)
                    break
                buf += data
                while len(buf) >= chunk_bytes:
                    raw = buf[:chunk_bytes]
                    buf = buf[chunk_bytes:]
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    yield samples
        except Exception as exc:
            log.error("Stream error: %s", exc)
        finally:
            try:
                proc.kill()
            except Exception:
                pass

        time.sleep(RECONNECT_DELAY_SECONDS)
