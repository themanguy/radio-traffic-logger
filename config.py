import os
from dotenv import load_dotenv

load_dotenv()

# Stream
STREAM_URL = os.getenv("STREAM_URL", "")

# Whisper
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

# Whisper prompt priming for NSW Fire and Rescue / NSW Ambulance shared channel
WHISPER_INITIAL_PROMPT = os.getenv(
    "WHISPER_INITIAL_PROMPT",
    (
        "Fire and Rescue NSW, FRNSW, NSW Ambulance. "
        "NSW Southern Tablelands region. "
        "Towns and villages: Goulburn, Queanbeyan, Cooma, Yass, Crookwell, "
        "Bungendore, Braidwood, Marulan, Taralga, Gunning, Collector, Tarago, "
        "Captains Flat, Michelago, Nimmitabel, Bombala, Delegate, Adaminaby, "
        "Berridale, Jindabyne, Tumut, Tumbarumba, Batlow, Cootamundra, Gundagai, "
        "Bowral, Moss Vale, Mittagong, Berrima, Bundanoon, Exeter, Sutton Forest, "
        "Marulan, Tallong, Wingello, Penrose, Robertson, Kangaroo Valley. "
        "Major roads: Hume Highway, Federal Highway, Kings Highway, Monaro Highway, "
        "Snowy Mountains Highway, Barton Highway, Olympic Highway, Illawarra Highway, "
        "Braidwood Road, Yass Road, Cooma Road. "
        "FRNSW callsigns: Goulburn 1, Goulburn 2, Queanbeyan 1, Cooma 1, Yass 1, "
        "Crookwell 1, Braidwood 1, Marulan 1, Tumut 1, Bowral 1, Moss Vale 1, "
        "Pump, Tanker, Rescue, breathing apparatus, BA, zone commander. "
        "NSW Ambulance callsigns: Intensive Care Paramedic, ICP, ALS, BLS, "
        "on road supervisor, ORS, Goulburn, Queanbeyan, Cooma, Yass. "
        "Dispatch terminology: structure fire, working fire, house fire, shed fire, "
        "grass fire, scrub fire, bushfire, ember attack, vehicle fire, MVA, "
        "road crash rescue, entrapment, rollover, semi-trailer, "
        "cardiac arrest, unconscious, difficulty breathing, chest pain, fall, "
        "Code 1, Code 2, on scene, staging, make pumps, persons reported, "
        "all persons accounted for, stop message, under control, loss stop, "
        "rural fire, RFS, property threatened, property destroyed, "
        "hazmat, chemical, fuel spill, stock on road, tree on road."
    ),
)

# VAD
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MIN_SPEECH_DURATION_MS = int(os.getenv("VAD_MIN_SPEECH_DURATION_MS", "250"))
VAD_MIN_SILENCE_DURATION_MS = int(os.getenv("VAD_MIN_SILENCE_DURATION_MS", "400"))
# Padding added before/after each speech segment (seconds)
VAD_SPEECH_PAD_SAMPLES = int(os.getenv("VAD_SPEECH_PAD_SAMPLES", "1600"))  # 0.1 s @ 16kHz

# Database
DB_PATH = os.getenv("DB_PATH", "transmissions.db")

# Keyword watchlist — comma-separated, case-insensitive
_kw = os.getenv("KEYWORD_WATCHLIST", "")
KEYWORD_WATCHLIST: list[str] = [k.strip().lower() for k in _kw.split(",") if k.strip()]

# Word blacklist — transcriptions containing any of these phrases are silently dropped.
# Covers common Whisper hallucinations on radio static/noise.
_default_blacklist = [
    # YouTube / podcast hallucinations
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "don't forget to subscribe",
    "see you next time",
    "see you in the next video",
    "in the next episode",
    "visit our website",
    "check out our",
    # Music / ambient noise hallucinations
    "♪",
    "♩",
    "la la la",
    "hmm hmm",
    # Things that would never appear on this channel
    "weather forecast",
    "stock market",
    "breaking news",
    "good morning everyone",
    "good evening everyone",
    "ladies and gentlemen",
    "welcome to",
    "thank you all",
    "i appreciate",
    "cryptocurrency",
    "bitcoin",
    "subscribe to",
]
_bw = os.getenv("WORD_BLACKLIST", "")
_extra = [w.strip().lower() for w in _bw.split(",") if w.strip()]
WORD_BLACKLIST: list[str] = [w.lower() for w in _default_blacklist] + _extra

# Slack
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# Reconnect
RECONNECT_DELAY_SECONDS = int(os.getenv("RECONNECT_DELAY_SECONDS", "5"))
