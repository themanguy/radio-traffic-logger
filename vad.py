"""
Voice Activity Detection using Silero VAD.
Consumes a stream of 512-sample float32 PCM chunks and yields complete
speech segments as concatenated numpy arrays.
"""

import numpy as np
import torch
import logging
from collections.abc import Iterator
from config import (
    VAD_THRESHOLD,
    VAD_MIN_SPEECH_DURATION_MS,
    VAD_MIN_SILENCE_DURATION_MS,
    VAD_SPEECH_PAD_SAMPLES,
)
from ingest import SAMPLE_RATE

log = logging.getLogger(__name__)


def load_vad_model():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        verbose=False,
    )
    return model


class SpeechSegmenter:
    """
    Stateful wrapper around Silero VAD that buffers incoming 512-sample chunks
    and emits complete speech segments.
    """

    def __init__(self, model):
        self.model = model
        self.sample_rate = SAMPLE_RATE
        self.threshold = VAD_THRESHOLD
        self.min_speech_samples = int(VAD_MIN_SPEECH_DURATION_MS * SAMPLE_RATE / 1000)
        self.min_silence_samples = int(VAD_MIN_SILENCE_DURATION_MS * SAMPLE_RATE / 1000)
        self.pad_samples = VAD_SPEECH_PAD_SAMPLES

        self._reset()

    def _reset(self):
        self.model.reset_states()
        self._speech_buf: list[np.ndarray] = []
        self._pre_buf: list[np.ndarray] = []  # rolling pre-speech context
        self._in_speech = False
        self._silence_samples = 0
        self._speech_samples = 0

    def process(self, chunks: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        """
        Feed PCM chunks, yield numpy arrays for each detected speech segment.
        Each yielded array is float32, shape (N,), at 16kHz.
        """
        pre_buf_max = self.pad_samples

        for chunk in chunks:
            tensor = torch.from_numpy(chunk)
            with torch.no_grad():
                prob = self.model(tensor, self.sample_rate).item()

            is_speech = prob >= self.threshold

            if is_speech:
                if not self._in_speech:
                    # Start of a new segment — prepend context buffer
                    self._in_speech = True
                    self._silence_samples = 0
                    # include pre-roll for context
                    pre = np.concatenate(self._pre_buf) if self._pre_buf else np.array([], dtype=np.float32)
                    self._speech_buf = [pre, chunk]
                else:
                    self._speech_buf.append(chunk)
                    self._silence_samples = 0
                self._speech_samples += len(chunk)
            else:
                if self._in_speech:
                    self._speech_buf.append(chunk)
                    self._silence_samples += len(chunk)

                    if self._silence_samples >= self.min_silence_samples:
                        # End of speech segment
                        segment = np.concatenate(self._speech_buf)
                        if self._speech_samples >= self.min_speech_samples:
                            yield segment
                        else:
                            log.debug("Skipping short segment (%d samples)", self._speech_samples)
                        self._in_speech = False
                        self._speech_samples = 0
                        self._silence_samples = 0
                        self._speech_buf = []
                        self._pre_buf = []

                # Maintain rolling pre-speech context
                self._pre_buf.append(chunk)
                total_pre = sum(len(c) for c in self._pre_buf)
                while total_pre - len(self._pre_buf[0]) >= pre_buf_max:
                    total_pre -= len(self._pre_buf.pop(0))
