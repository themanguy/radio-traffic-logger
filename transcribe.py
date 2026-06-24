"""
Transcription via faster-whisper.
"""

import numpy as np
import logging
from dataclasses import dataclass
from faster_whisper import WhisperModel
from config import WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_INITIAL_PROMPT
from ingest import SAMPLE_RATE

log = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    duration_seconds: float
    avg_confidence: float  # average log-prob converted to [0,1]
    language: str


def load_whisper(model_size: str = WHISPER_MODEL) -> WhisperModel:
    log.info("Loading Whisper model: %s", model_size)
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe(model: WhisperModel, audio: np.ndarray) -> TranscriptionResult | None:
    """
    Transcribe a float32 numpy array (16kHz mono).
    Returns None if no speech was recognised.
    """
    duration = len(audio) / SAMPLE_RATE

    segments, info = model.transcribe(
        audio,
        language=WHISPER_LANGUAGE,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        vad_filter=False,  # we handle VAD ourselves
        word_timestamps=False,
        beam_size=5,
    )

    texts = []
    log_probs = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            texts.append(text)
            if seg.avg_logprob is not None:
                # avg_logprob is negative; convert to rough confidence [0,1]
                log_probs.append(min(1.0, max(0.0, 1.0 + seg.avg_logprob / 10.0)))

    if not texts:
        return None

    return TranscriptionResult(
        text=" ".join(texts),
        duration_seconds=duration,
        avg_confidence=float(np.mean(log_probs)) if log_probs else 0.0,
        language=info.language,
    )
