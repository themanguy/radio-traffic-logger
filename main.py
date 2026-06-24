"""
Radio Traffic Logger — main entry point.

Usage:
    python main.py [--stream URL] [--model small|medium] [--export-csv]
"""

import argparse
import logging
import os
import signal
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

import config
from ingest import stream_pcm
from vad import load_vad_model, SpeechSegmenter
from transcribe import load_whisper, transcribe
from database import init_db, insert_transmission, export_csv

# Slack notifications (optional — silent if webhook not configured)
sys.path.insert(0, os.path.expanduser("~/projects/shared"))
from slack_notify import notify as _slack_notify

def _slack(title, message="", level="info", fields=None):
    _slack_notify(title=title, message=message, level=level,
                  channel="fenz", fields=fields, source="fenz-monitor")

# ── logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# ── ANSI colours ───────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"


def _check_keywords(text: str) -> list[str]:
    lower = text.lower()
    return [kw for kw in config.KEYWORD_WATCHLIST if kw in lower]


def _is_blacklisted(text: str) -> bool:
    lower = text.lower().strip()
    # Drop single-word / very short hallucinations
    if len(lower.split()) <= 1 and lower not in {"mayday", "fire", "help"}:
        return True
    return any(phrase in lower for phrase in config.WORD_BLACKLIST)


def _print_transmission(
    ts: datetime,
    text: str,
    duration: float,
    confidence: float,
    matched_kw: list[str],
    row_id: int,
) -> None:
    flag = f"  {RED}⚑ KEYWORDS: {', '.join(matched_kw)}{RESET}" if matched_kw else ""
    conf_color = GREEN if confidence >= 0.7 else YELLOW if confidence >= 0.4 else RED
    print(
        f"\n{BOLD}{CYAN}[{ts.strftime('%H:%M:%S')}]{RESET}  "
        f"#{row_id}  {duration:.1f}s  "
        f"{conf_color}conf={confidence:.2f}{RESET}"
        f"{flag}\n"
        f"  {text}"
    )


def run(stream_url: str, model_size: str) -> None:
    if not stream_url:
        log.error("No stream URL configured. Set STREAM_URL in .env or pass --stream.")
        sys.exit(1)

    log.info("Initialising models...")
    vad_model  = load_vad_model()
    whisper    = load_whisper(model_size)
    db         = init_db()
    segmenter  = SpeechSegmenter(vad_model)

    log.info("Pipeline ready. Listening to: %s", stream_url)
    log.info("Keyword watchlist: %s", config.KEYWORD_WATCHLIST or "(none)")
    _slack(
        "📡 FENZ Monitor Started",
        message=f"Listening to stream.",
        fields={
            "Stream": stream_url[:60],
            "Whisper model": model_size,
            "Keywords": ", ".join(config.KEYWORD_WATCHLIST) or "(none)",
        },
    )

    # Graceful shutdown on Ctrl-C
    def _shutdown(sig, frame):
        print(f"\n{YELLOW}Shutting down…{RESET}")
        db.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    try:
        pcm_stream = stream_pcm(stream_url)
    except Exception as e:
        log.error("Failed to open stream: %s", e)
        _slack("🔴 FENZ: Stream Error", message=str(e), level="error",
               fields={"Stream": stream_url[:60]})
        sys.exit(1)

    for segment in segmenter.process(pcm_stream):
        result = transcribe(whisper, segment)
        if result is None or not result.text.strip():
            continue
        if _is_blacklisted(result.text):
            log.debug("Blacklisted: %s", result.text)
            continue

        now     = datetime.now(timezone.utc)
        matched = _check_keywords(result.text)
        row_id  = insert_transmission(
            db,
            transcript=result.text,
            duration_seconds=result.duration_seconds,
            confidence=result.avg_confidence,
            flagged=bool(matched),
            keywords_matched=matched,
            timestamp=now,
        )
        _print_transmission(now, result.text, result.duration_seconds,
                            result.avg_confidence, matched, row_id)

        if matched:
            _slack(
                "🚨 FENZ: Keyword Detected",
                message=result.text[:200],
                level="warning",
                fields={
                    "Keywords":  ", ".join(matched),
                    "Timestamp": now.strftime("%H:%M:%S UTC"),
                    "Duration":  f"{result.duration_seconds:.1f}s",
                    "Confidence": f"{result.avg_confidence:.2f}",
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Radio Traffic Logger")
    parser.add_argument("--stream", default=config.STREAM_URL,
                        help="HTTP audio stream URL")
    parser.add_argument("--model", default=config.WHISPER_MODEL,
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export DB to CSV and exit")
    args = parser.parse_args()

    if args.export_csv:
        out = export_csv()
        print(f"Exported to: {out}")
        return

    run(stream_url=args.stream, model_size=args.model)


if __name__ == "__main__":
    main()
