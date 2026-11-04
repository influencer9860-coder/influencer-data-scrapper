from __future__ import annotations

import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import logging
import os
import re
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from email_validator import EmailNotValidError, validate_email
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from rapidocr_onnxruntime import RapidOCR
from supabase import create_client
from telegram import Update
from telegram.constants import ChatType
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

load_dotenv()

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("hpack").setLevel(logging.WARNING)
log = logging.getLogger("creator-bot")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

DAILY_TARGET = int(os.getenv("DAILY_TARGET", "50"))
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "15"))
ADMIN_ID = int(os.getenv("AUTHORIZED_USER_ID", "0"))

# Hardware & Concurrency Settings
CPU_COUNT = os.cpu_count() or 4
OCR_WORKERS = max(1, int(os.getenv("OCR_WORKERS", str(max(1, min(4, CPU_COUNT // 2))))))
OCR_THREADS = max(1, int(os.getenv("OCR_THREADS", str(max(1, CPU_COUNT // OCR_WORKERS)))))
IO_EXTRA_WORKERS = 2
TARGET_OCR_WIDTH = int(os.getenv("OCR_WIDTH", "720"))
PROGRESS_EVERY = float(os.getenv("PROGRESS_EVERY", "2.0"))
BATCH_IDLE_SECS = float(os.getenv("BATCH_IDLE_SECS", "2.5"))
BATCH_CHUNK_SIZE = int(os.getenv("BATCH_CHUNK_SIZE", "25"))  # Safe execution window

DB_MAX_CONCURRENCY = max(1, int(os.getenv("DB_MAX_CONCURRENCY", "1")))
DB_RETRIES = max(1, int(os.getenv("DB_RETRIES", "5")))

try:
    TZ = ZoneInfo(os.getenv("TIMEZONE", "Asia/Kolkata"))
except Exception as exc:
    raise SystemExit(f"Invalid TIMEZONE: {exc}")

if not all([SUPABASE_URL, SUPABASE_KEY, BOT_TOKEN]) or ADMIN_ID <= 0:
    raise SystemExit("Missing required .env settings: SUPABASE_URL, SUPABASE_KEY, TELEGRAM_BOT_TOKEN, AUTHORIZED_USER_ID")

# -----------------------------------------------------------------------------
# THREAD-SAFE SUPABASE CONNECTION
# -----------------------------------------------------------------------------
log.info("Connecting to Supabase at %s", SUPABASE_URL)
_db_local = threading.local()
_db_generation = 0
_db_gen_lock = threading.Lock()


def _bump_db_generation() -> None:
    global _db_generation
    with _db_gen_lock:
        _db_generation += 1


def get_db():
    client = getattr(_db_local, "client", None)
    if client is None or getattr(_db_local, "gen", -1) != _db_generation:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        _db_local.client = client
        _db_local.gen = _db_generation
    return client


class _DBProxy:
    def __getattr__(self, name):
        return getattr(get_db(), name)


db = _DBProxy()
get_db()

_db_sem: asyncio.Semaphore | None = None


def _get_db_sem() -> asyncio.Semaphore:
    global _db_sem
    if _db_sem is None:
        _db_sem = asyncio.Semaphore(DB_MAX_CONCURRENCY)
    return _db_sem


def _is_transient_db_error(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    text = str(exc).lower()
    markers = (
        "json could not be generated", "bad request", "<html", "502", "503", "504",
        "gateway", "timed out", "timeout", "connection", "reset by peer", "remote protocol",
        "server disconnected", "temporarily unavailable",
    )
    return any(m in text for m in markers)


async def run_db(fn):
    last_exc: Exception | None = None
    for attempt in range(DB_RETRIES):
        try:
            async with _get_db_sem():
                return await asyncio.to_thread(fn)
        except Exception as exc:
            last_exc = exc
            if attempt < DB_RETRIES - 1 and _is_transient_db_error(exc):
                log.warning("[DB RETRY %d/%d] Transient issue: %s", attempt + 1, DB_RETRIES, str(exc)[:200])
                _bump_db_generation()
                await asyncio.sleep(0.4 * (attempt + 1))
                continue
            raise
    raise last_exc


# -----------------------------------------------------------------------------
# RAPIDOCR ENGINE INITIALIZER & WARMUP
# -----------------------------------------------------------------------------
_tl = threading.local()


def _make_engine():
    try:
        return RapidOCR(intra_op_num_threads=OCR_THREADS)
    except Exception:
        return RapidOCR()


def get_engine():
    eng = getattr(_tl, "engine", None)
    if eng is None:
        log.info("Loading RapidOCR engine (ONNX Runtime) in thread %s...", threading.current_thread().name)
        eng = _make_engine()
        _tl.engine = eng
        log.info("RapidOCR initialized successfully in thread %s.", threading.current_thread().name)
    return eng


OCR_POOL = ThreadPoolExecutor(max_workers=OCR_WORKERS, thread_name_prefix="ocr", initializer=get_engine)


def _warmup_task(barrier: threading.Barrier):
    try:
        img = Image.new("RGB", (320, 320), "white")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        get_engine()(buf.getvalue())
    except Exception:
        log.debug("Warm-up OCR failed (non-fatal)", exc_info=True)
    try:
        barrier.wait(timeout=180)
    except threading.BrokenBarrierError:
        pass


async def warm_up_engines():
    log.info("Warming up %d OCR worker(s) x %d ONNX thread(s)...", OCR_WORKERS, OCR_THREADS)
    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    barrier = threading.Barrier(OCR_WORKERS)
    await asyncio.gather(*[loop.run_in_executor(OCR_POOL, _warmup_task, barrier) for _ in range(OCR_WORKERS)])
    log.info("OCR engines ready in %.1fs", time.perf_counter() - t0)


# -----------------------------------------------------------------------------
# TAXONOMY DICTIONARIES
# -----------------------------------------------------------------------------
IATA_AIRPORT_CODES = {
    "DEL": "Delhi, India", "BOM": "Mumbai, India", "BLR": "Bengaluru, India",
    "HYD": "Hyderabad, India", "MAA": "Chennai, India", "CCU": "Kolkata, India",
    "PNQ": "Pune, India", "AMD": "Ahmedabad, India", "GOI": "Goa, India",
    "DXB": "Dubai, UAE", "LHR": "London, UK", "JFK": "New York, USA",
    "SFO": "San Francisco, USA", "SIN": "Singapore", "YYZ": "Toronto, Canada",
}

INDIAN_CITIES = {
    "mumbai": "Mumbai, India", "bombay": "Mumbai, India", "navi mumbai": "Navi Mumbai, India",
    "thane": "Thane, India", "pune": "Pune, India", "delhi": "Delhi, India",
    "new delhi": "New Delhi, India", "dilli": "Delhi, India", "noida": "Noida, India",
    "greater noida": "Greater Noida, India", "gurgaon": "Gurugram, India",
    "gurugram": "Gurugram, India", "faridabad": "Faridabad, India", "ghaziabad": "Ghaziabad, India",
    "bangalore": "Bengaluru, India", "bengaluru": "Bengaluru, India", "hyderabad": "Hyderabad, India",
    "secunderabad": "Hyderabad, India", "chennai": "Chennai, India", "madras": "Chennai, India",
    "kolkata": "Kolkata, India", "calcutta": "Kolkata, India", "ahmedabad": "Ahmedabad, India",
    "surat": "Surat, India", "vadodara": "Vadodara, India", "baroda": "Vadodara, India",
    "rajkot": "Rajkot, India", "jaipur": "Jaipur, India", "jodhpur": "Jodhpur, India",
    "udaipur": "Udaipur, India", "kota": "Kota, India", "lucknow": "Lucknow, India",
    "kanpur": "Kanpur, India", "varanasi": "Varanasi, India", "banaras": "Varanasi, India",
    "prayagraj": "Prayagraj, India", "allahabad": "Prayagraj, India", "agra": "Agra, India",
    "indore": "Indore, India", "bhopal": "Bhopal, India", "gwalior": "Gwalior, India",
    "jabalpur": "Jabalpur, India", "chandigarh": "Chandigarh, India", "mohali": "Mohali, India",
    "panchkula": "Panchkula, India", "ludhiana": "Ludhiana, India", "amritsar": "Amritsar, India",
    "jalandhar": "Jalandhar, India", "patna": "Patna, India", "ranchi": "Ranchi, India",
    "jamshedpur": "Jamshedpur, India", "bhubaneswar": "Bhubaneswar, India", "cuttack": "Cuttack, India",
    "guwahati": "Guwahati, India", "shillong": "Shillong, India", "nagpur": "Nagpur, India",
    "nashik": "Nashik, India", "aurangabad": "Chhatrapati Sambhajinagar, India",
    "kolhapur": "Kolhapur, India", "goa": "Goa, India", "panaji": "Goa, India",
    "kochi": "Kochi, India", "cochin": "Kochi, India", "trivandrum": "Thiruvananthapuram, India",
    "thiruvananthapuram": "Thiruvananthapuram, India", "calicut": "Kozhikode, India",
    "kozhikode": "Kozhikode, India", "coimbatore": "Coimbatore, India", "madurai": "Madurai, India",
    "mysore": "Mysuru, India", "mysuru": "Mysuru, India", "mangalore": "Mangaluru, India",
    "mangaluru": "Mangaluru, India", "visakhapatnam": "Visakhapatnam, India", "vizag": "Visakhapatnam, India",
    "vijayawada": "Vijayawada, India", "dehradun": "Dehradun, India", "rishikesh": "Rishikesh, India",
    "shimla": "Shimla, India", "manali": "Manali, India", "srinagar": "Srinagar, India",
    "jammu": "Jammu, India", "raipur": "Raipur, India", "dhanbad": "Dhanbad, India",
}

INDIAN_STATES = {
    "odisha": "Odisha, India", "orissa": "Odisha, India", "maharashtra": "Maharashtra, India",
    "karnataka": "Karnataka, India", "tamil nadu": "Tamil Nadu, India", "kerala": "Kerala, India",
    "gujarat": "Gujarat, India", "rajasthan": "Rajasthan, India", "punjab": "Punjab, India",
    "haryana": "Haryana, India", "uttar pradesh": "Uttar Pradesh, India", "madhya pradesh": "Madhya Pradesh, India",
    "bihar": "Bihar, India", "west bengal": "West Bengal, India", "assam": "Assam, India",
    "himachal pradesh": "Himachal Pradesh, India", "uttarakhand": "Uttarakhand, India",
    "telangana": "Telangana, India", "andhra pradesh": "Andhra Pradesh, India",
    "goa": "Goa, India", "kashmir": "Jammu & Kashmir, India",
}

GLOBAL_LOCATIONS = {
    "chicago": "Chicago, USA", "cambridge": "Cambridge, UK", "london": "London, UK",
    "manchester": "Manchester, UK", "dubai": "Dubai, UAE", "abu dhabi": "Abu Dhabi, UAE",
    "sharjah": "Sharjah, UAE", "new york": "New York, USA", "nyc": "New York, USA",
    "los angeles": "Los Angeles, USA", "la": "Los Angeles, USA", "san francisco": "San Francisco, USA",
    "toronto": "Toronto, Canada", "vancouver": "Vancouver, Canada", "singapore": "Singapore",
    "bali": "Bali, Indonesia", "sydney": "Sydney, Australia", "melbourne": "Melbourne, Australia",
}

NICHE_RULES = {
    "Fashion & Style": (
        r"\b(fashion|style|stylist|ootd|wardrobe|clothing|outfits?|couture|ethnic|"
        r"hauls?|saree|dresses|zara|h&m|lookbook|streetwear|menswear|womenswear)\b"
    ),
    "Beauty & Skincare": (
        r"\b(beauty|skincare|skin\s*care|makeup|mua|cosmetics|haircare|hair\s*care|"
        r"dermat|glow|aesthetics|lipstick|fragrance|perfume)\b"
    ),
    "Lifestyle": (
        r"\b(lifestyle|daily\s*vlogs?|vlogger|vlogs?|day\s*in\s*my\s*life|living|"
        r"aesthetic|lifestyleblogger|routine|life\s*updates?)\b"
    ),
    "UGC & Creator Economy": (
        r"\b(ugc|ugc\s*creator|content\s*creator|digital\s*creator|video\s*creator|"
        r"reel\s*creator|influencer|pr\s*friendly|brand\s*collabs?|spill\s*drish)\b"
    ),
    "Fitness & Nutrition": (
        r"\b(fitness|gym|workout|health|wellness|nutrition|trainer|yoga|bodybuilding|"
        r"calisthenics|crossfit|running|athlete|physique|fitfam|weightloss)\b"
    ),
    "Food & Culinary": (
        r"\b(food|foodie|recipes?|cooking|cook|chef|baker|baking|cafe|street\s*food|"
        r"restaurants?|gourmet|vegetarian|vegan|kitchen|dining|delish)\b"
    ),
    "Travel & Hospitality": (
        r"\b(travel|traveler|traveller|wanderlust|explorer|backpacking|hotels?|resorts?|"
        r"mountains|beaches|itinerary|nomad|vacation|trips?)\b"
    ),
    "Tech & Gadgets": (
        r"\b(tech|technology|coding|developer|software|gadgets?|ai|artificial\s*intelligence|"
        r"machine\s*learning|unboxing|tech\s*reviews?|cyber|web3|crypto)\b"
    ),
    "Business & Finance": (
        r"\b(business|finance|entrepreneur|founder|ceo|startup|investing|investor|"
        r"stocks?|trading|money|real\s*estate|marketing|consultant)\b"
    ),
    "Parenting & Family": (
        r"\b(mom|motherhood|parenting|dad|family|baby|toddler|kids|pregnancy|momblogger)\b"
    ),
    "Entertainment & Comedy": (
        r"\b(comedy|comedian|standup|humor|actor|acting|theatre|cinema|memes?|entertainment)\b"
    ),
    "Dance & Performance": (
        r"\b(dance|dancer|choreographer|choreography|zumba|classical\s*dance|hiphop)\b"
    ),
    "Art & Photography": (
        r"\b(artist|art|painting|sketching|photographer|photography|filmmaker|cinematography|designer|design|architect)\b"
    ),
    "Modeling & Editorial": (
        r"\b(model|modeling|editorial|runway|pageant|commercial\s*model|photoshoots?)\b"
    ),
    "Education & Books": (
        r"\b(educator|teacher|study|books?|bookstagram|reader|literature|coaching|motivation|mentor)\b"
    ),
}

# Complete Unicode Emoji and Dingbat Filter
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\u2600-\u27BF"
    "\uFE0E-\uFE0F"
    "\u200D"
    "]+",
    flags=re.UNICODE,
)

PRONOUN_RE = re.compile(
    r"(?i)\b(?:she|he|they|her|him|them|his|hers|its|xe|ze)\s*/\s*(?:she|he|they|her|him|them|his|hers|its|xe|ze)(?:\s*/\s*\w+)?\b"
)

VALID_TLD_RE = re.compile(
    r"\.(com|in|co\.in|org|net|io|ai|me|co|app|agency|media|info|biz|tech|store|xyz|live|tv|uk|us|ca|ae|de|fr|global)$",
    re.I,
)

EMAIL_RE = re.compile(r"(?i)(?<![A-Z0-9._%+-])[A-Z0-9][A-Z0-9._%+-]*@[A-Z0-9.-]+\.[A-Z]{2,}(?![A-Z0-9.-])")
HANDLE_RE = re.compile(r"^[A-Za-z0-9._]{2,30}$")
COUNT_TOKEN_RE = re.compile(r"(?i)(?<![A-Za-z0-9])([0-9][0-9,.]*\s*[KMB]?)(?![A-Za-z])")
NUM_ONLY_RE = re.compile(r"^[0-9][0-9,.\s]*[KMBkmb]?$")

DOMAIN_TYPOS = {
    "gmai.com": "gmail.com", "gmial.com": "gmail.com", "gamil.com": "gmail.com",
    "gmaill.com": "gmail.com", "gmai1.com": "gmail.com", "yaho.com": "yahoo.com",
    "yahou.com": "yahoo.com", "outlok.com": "outlook.com", "hotmial.com": "hotmail.com", "icld.com": "icloud.com",
}

PROVIDER_DOMAINS = sorted(
    [
        "gmail.com", "googlemail.com", "yahoo.com", "yahoo.in", "yahoo.co.in", "outlook.com",
        "hotmail.com", "icloud.com", "live.com", "rediffmail.com", "proton.me", "protonmail.com", "me.com",
    ],
    key=len,
    reverse=True,
)
_TRIM_CHARS = "<>()[]{}.,;:!?'\"-–—|•·¦/\u2502\u2503 "


def repair_missing_at(token: str) -> str | None:
    if not token or "@" in token:
        return None
    t = token.strip(_TRIM_CHARS)
    low = t.lower()
    for dom in PROVIDER_DOMAINS:
        if not low.endswith(dom):
            continue
        local = t[: -len(dom)]
        if len(local) < 3:
            continue
        last, prev = local[-1], local[-2]
        junk_symbol = (not last.isalnum()) and last not in "._-"
        junk_capital = last.isupper() and (prev.islower() or prev.isdigit())
        junk_six = last == "6" and (prev.isalnum() or prev in "._")
        if (junk_symbol or junk_capital or junk_six) and len(local) - 1 >= 3:
            local = local[:-1]
        local = local.rstrip("._-")
        if len(local) < 2:
            continue
        return f"{local}@{dom}"
    return None


def now_local() -> datetime:
    return datetime.now(TZ)


def parse_count(value: str | None) -> int | None:
    if not value:
        return None
    match = COUNT_TOKEN_RE.search(str(value).replace("\u00a0", " "))
    if not match:
        return None
    token = re.sub(r"\s+", "", match.group(1)).upper()
    suffix = token[-1] if token[-1:] in {"K", "M", "B"} else ""
    number = token[:-1] if suffix else token
    if not suffix and re.fullmatch(r"\d{1,3}([.,]\d{3})+", number):
        number = re.sub(r"[.,]", "", number)
    else:
        number = number.replace(",", "")
    try:
        multiplier = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
        result = int(float(number) * multiplier)
        return result if result >= 0 else None
    except (ValueError, OverflowError):
        return None


# ----------------------------------------------------------------- Entity Extraction
def clean_display_name(raw_name: str) -> tuple[str, str | None]:
    """Cleans emojis, bio subtitles, underscores, trailing quotes, and clamps to max 2 words."""
    clean = EMOJI_PATTERN.sub(" ", raw_name)
    clean = PRONOUN_RE.sub(" ", clean)
    clean = re.sub(r"('s)([A-Za-z])", r"\1 \2", clean)

    # Split on subtitle delimiters
    parts = re.split(r"\s*[|•·/–—\-~]\s*", clean.strip())
    name_candidate = parts[0].strip()
    tag = parts[1].strip() if len(parts) > 1 and len(parts[1].strip()) >= 3 else None

    # Replace glued underscores: Bhargavi_Vegiraju -> Bhargavi Vegiraju
    name_candidate = name_candidate.replace("_", " ")

    # Strip OCR punctuation/quotes (like the trailing apostrophe in Purvi Khirani')
    name_candidate = re.sub(r"[^A-Za-zÀ-ÿ\s'-]", " ", name_candidate)
    name_candidate = re.sub(r"\s+", " ", name_candidate).strip(" '\"-–—*.,;:!?`")

    # Split CamelCase if glued together by OCR
    if " " not in name_candidate and len(re.findall(r"(?<=[a-z])(?=[A-Z])", name_candidate)) == 1:
        name_candidate = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name_candidate)

    # Strip trailing quotes from individual tokens
    raw_words = [w.strip(" '\"-–—*.,;:!?`") for w in name_candidate.split()]
    words = [w for w in raw_words if len(w) > 0]

    # Enforce strict 2-word maximum
    if len(words) > 2:
        final_name = f"{words[0]} {words[1]}"
    elif len(words) >= 1:
        final_name = " ".join(words)
    else:
        final_name = "UNKNOWN"

    return (final_name.title() if len(final_name) >= 2 else "UNKNOWN", tag)


class EmailEngine:
    @staticmethod
    def clean(raw_candidate: str, matched_username: str | None = None) -> str | None:
        if not raw_candidate or "@" not in raw_candidate:
            return None

        candidate = re.split(r"\.{2,}|…", raw_candidate)[0]
        candidate = re.sub(r"(?i)(\.(?:com|in|co\.in|org|net|io|ai|me|co|app|agency|media))(?:\s*(?:and\s*\d+\s*more|more).*)$", r"\1", candidate)
        candidate = candidate.replace("(at)", "@").replace("[at]", "@")
        candidate = re.sub(r"\s+", "", candidate).strip(_TRIM_CHARS)

        match = EMAIL_RE.search(candidate)
        if not match:
            return None

        email = match.group(0).lower().rstrip(".,;:-–—|")
        if "@" not in email:
            return None

        local_part, domain = email.split("@", 1)

        # 1. Strip common prefix words
        local_part = re.sub(
            r"(?i)^.*?(?:collabs?|collaborat(?:e|ions?)|paid|pr|work|business|biz|enquir(?:y|ies)|queries|query|bookings?|email|mail|dm|contact|reach|connect|for)[-_:/|~]+\s*",
            "", local_part
        )
        # 2. Strip leading punctuation / symbols / envelope glyph remnants
        local_part = re.sub(r"^[|¦!\\/:\-~._•·]+", "", local_part)

        # 3. Strip OCR ghost prefix matching username (e.g. lyourbeautsquad -> yourbeautsquad)
        if matched_username:
            u_clean = re.sub(r"[._]", "", matched_username.lower())
            if u_clean and len(u_clean) >= 4:
                for prefix in ("l", "i", "1", "|", "¦", "•"):
                    if local_part.startswith(prefix) and (
                        local_part[len(prefix):].startswith(u_clean) or local_part[len(prefix):] == u_clean
                    ):
                        log.info("  [EMAIL SCRUB] Stripped ghost prefix '%s' matching @%s", prefix, matched_username)
                        local_part = local_part[len(prefix):]
                        break

        local_part = re.sub(r"^[^a-z0-9]+", "", local_part)

        if domain in DOMAIN_TYPOS:
            domain = DOMAIN_TYPOS[domain]

        if not VALID_TLD_RE.search(domain):
            return None

        if domain in {"gmail.com", "googlemail.com"} and len(local_part) < 6:
            return None
        if len(local_part) < 2:
            return None

        repaired_email = f"{local_part}@{domain}"
        try:
            return validate_email(repaired_email, check_deliverability=False).normalized.lower()
        except EmailNotValidError:
            return None
        except Exception:
            return repaired_email if "@" in repaired_email and "." in domain else None


class GeoLocator:
    @staticmethod
    def _clean_token(t: str) -> str:
        t = re.sub(r"^[^a-zA-Z0-9]+", "", t)
        t = re.sub(r"[^a-zA-Z0-9]+$", "", t)
        if len(t) > 4 and t[0] in "PpQq09" and t[1].isupper():
            t = t[1:]
        return t

    @classmethod
    def detect(cls, bio_text: str) -> str | None:
        """Strictly detects location from the bio section only. Never scans highlight circles or post grids."""
        if not bio_text:
            return None

        lines = bio_text.splitlines()
        # High confidence: lines explicitly accompanied by pin or location keywords
        pin_lines = [
            l for l in lines
            if any(marker in l for marker in ["📍", "📌", "location", "based in", "living in"])
        ]

        def match_tables(target_text: str) -> str | None:
            if not target_text:
                return None
            spaced = re.sub(r"[/|•·\n,–—~]+", " ", target_text)
            cleaned = EMOJI_PATTERN.sub(" ", spaced)
            tokens = [cls._clean_token(t) for t in cleaned.split() if cls._clean_token(t)]
            clean_sentence = " ".join(tokens)
            clean_lower = clean_sentence.lower()

            # Word boundary matching on Indian Cities
            for city_key, canonical in INDIAN_CITIES.items():
                if re.search(r"\b" + re.escape(city_key) + r"\b", clean_lower):
                    return canonical

            # Global Hubs (e.g., Chicago, London, Dubai)
            for glob_key, canonical in GLOBAL_LOCATIONS.items():
                if re.search(r"\b" + re.escape(glob_key) + r"\b", clean_lower):
                    return canonical

            # Indian States (e.g., Odisha, Maharashtra)
            for state_key, canonical in INDIAN_STATES.items():
                if re.search(r"\b" + re.escape(state_key) + r"\b", clean_lower):
                    return canonical

            # IATA Codes
            for code, canonical in IATA_AIRPORT_CODES.items():
                if re.search(r"\b" + re.escape(code) + r"\b", clean_sentence):
                    return canonical

            # Glued names (e.g. WestBengal)
            glued = re.sub(r"[^a-z]", "", clean_lower)
            for table in (INDIAN_CITIES, GLOBAL_LOCATIONS, INDIAN_STATES):
                for key, canonical in table.items():
                    if " " in key and key.replace(" ", "") in glued:
                        return canonical
            return None

        # Check pin lines first
        if pin_lines:
            pin_match = match_tables(" \n ".join(pin_lines))
            if pin_match:
                return pin_match

        # Otherwise check the rest of the bio lines
        return match_tables(bio_text)


class NicheClassifier:
    @staticmethod
    def detect(text: str, subtitle: str | None = None, name_tag: str | None = None) -> str | None:
        detected = set()
        combined_text = f"{subtitle or ''} \n {name_tag or ''} \n {text}".strip()
        if not combined_text:
            return None

        for category, pattern in NICHE_RULES.items():
            if re.search(pattern, combined_text, re.I):
                detected.add(category)

        if not detected:
            if subtitle and len(subtitle.strip()) > 3:
                return subtitle.strip().title()
            return None

        return ", ".join(sorted(detected))


def _email_candidates_from_lines(all_lines: list[str], username: str | None = None) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    email_indicator_pattern = re.compile(r"(?:✉|📧|📩|email|mail|contact|collab|queries|pr)", re.I)

    for line in all_lines:
        has_indicator = bool(email_indicator_pattern.search(line))
        norm = re.sub(r"[|¦•·/\\]", " ", line)
        norm = re.sub(r"@\s+", "@", norm)
        norm = re.sub(r"([a-zA-Z0-9_.-]+)\s*\.\s*([a-zA-Z0-9_.-]+)", r"\1.\2", norm)

        for token in norm.split():
            if token.startswith("@"):
                continue

            tries = [(token, 0)]
            if "@" not in token:
                fixed = repair_missing_at(token)
                if fixed:
                    tries.append((fixed, 0))

            for cand, bonus in tries:
                e = EmailEngine.clean(cand, matched_username=username)
                if e:
                    score = (100 if has_indicator else 50) + bonus
                    if any(provider in e for provider in ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"]):
                        score += 20
                    candidates.append((score, e))
                    break
    return candidates


def _extract_from_boxes(boxes_data: list, full_w: int, full_h: int) -> dict:
    items = []
    for item in boxes_data:
        box, text, score = item[0], item[1].strip(), float(item[2])
        if not text or score < 0.25:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        items.append({
            "text": text,
            "score": score,
            "x1": x1 / full_w, "y1": y1 / full_h,
            "x2": x2 / full_w, "y2": y2 / full_h,
            "cx": ((x1 + x2) / 2.0) / full_w,
            "cy": ((y1 + y2) / 2.0) / full_h,
            "h": max(1.0, y2 - y1) / full_h,
            "w": max(1.0, x2 - x1) / full_w,
        })

    filtered_items = [
        it for it in items
        if it["cy"] >= 0.045 and not re.fullmatch(r"\d{1,2}:\d{2}", it["text"].strip())
    ]

    button_y = 0.55
    action_buttons = [
        it["cy"] for it in filtered_items
        if it["text"].lower() in {"follow", "following", "message", "edit profile", "contact", "share profile", "subscribe"} and it["cy"] > 0.25
    ]
    if action_buttons:
        button_y = min(action_buttons) + 0.02

    # Bio section items strictly above action buttons
    profile_items = [it for it in filtered_items if it["cy"] <= button_y]

    # 1. USERNAME
    blocked = {"instagram", "threads", "follow", "message", "posts", "followers", "following"}
    username = None
    header_candidates = []

    for it in profile_items:
        if 0.048 <= it["cy"] <= 0.140:
            clean_line = re.sub(r"[<«‹>»…•·*✓✔]+", " ", it["text"])
            for tok in clean_line.split():
                t = re.sub(r"^[^a-zA-Z0-9_]+", "", tok)
                t = re.sub(r"[^a-zA-Z0-9_]+$", "", t)
                if len(t) >= 2 and HANDLE_RE.fullmatch(t) and re.search(r"[A-Za-z]", t) and t.lower() not in blocked:
                    header_candidates.append((it["score"], t.lower()))

    if header_candidates:
        header_candidates.sort(key=lambda x: x[0], reverse=True)
        username = header_candidates[0][1]

    # 2. STATS
    labels = {
        "post_count": (re.compile(r"^posts?$", re.I), re.compile(r"\bposts?\b", re.I)),
        "followers": (re.compile(r"^followers?$", re.I), re.compile(r"\bfollowers?\b", re.I)),
        "following": (re.compile(r"^following$", re.I), re.compile(r"\bfollowing\b", re.I)),
    }
    stats = {"post_count": None, "followers": None, "following": None}
    stats_cys = []

    for key, (exact, loose) in labels.items():
        exact_boxes = [x for x in profile_items if exact.fullmatch(x["text"].strip()) and 0.14 <= x["cy"] <= 0.35]
        if exact_boxes:
            label_box = min(exact_boxes, key=lambda x: x["cy"])
            stats_cys.append(label_box["cy"])
            number_candidates = []
            for it in profile_items:
                if it is label_box:
                    continue
                gap = label_box["cy"] - it["cy"]
                dx = abs(label_box["cx"] - it["cx"])
                if 0.005 <= gap <= 0.065 and dx <= 0.10 and NUM_ONLY_RE.fullmatch(it["text"].strip()):
                    c = parse_count(it["text"])
                    if c is not None:
                        number_candidates.append((dx + gap * 0.5, c))
            if number_candidates:
                stats[key] = min(number_candidates)[1]
                continue

        merged = [x for x in profile_items if loose.search(x["text"]) and 0.14 <= x["cy"] <= 0.35]
        for box in sorted(merged, key=lambda x: x["cy"]):
            m = COUNT_TOKEN_RE.search(box["text"])
            if m:
                c = parse_count(m.group(1))
                if c is not None:
                    stats[key] = c
                    stats_cys.append(box["cy"])
                    break

    min_stat_cy = min(stats_cys) if stats_cys else 0.22
    if sum(v is not None for v in stats.values()) < 2:
        stat_row_numbers = [
            it for it in profile_items
            if (min_stat_cy - 0.05) <= it["cy"] <= (min_stat_cy + 0.02) and NUM_ONLY_RE.fullmatch(it["text"].strip())
        ]
        stat_row_numbers.sort(key=lambda x: x["cx"])
        if len(stat_row_numbers) == 3:
            stats["post_count"] = parse_count(stat_row_numbers[0]["text"])
            stats["followers"] = parse_count(stat_row_numbers[1]["text"])
            stats["following"] = parse_count(stat_row_numbers[2]["text"])

    # 3. DISPLAY NAME (Max 2 words)
    name_candidates = []
    for it in profile_items:
        clean_text = it["text"].strip()
        if not (0.10 <= it["cy"] <= min_stat_cy - 0.012):
            continue
        if it["x1"] < 0.20 or it["score"] < 0.30:
            continue
        if len(clean_text) < 2 or len(clean_text) > 60:
            continue
        if NUM_ONLY_RE.fullmatch(clean_text) or re.search(r"@|https?://|\.com\b", clean_text, re.I):
            continue
        if not re.search(r"[A-Za-zÀ-ÿ]", clean_text):
            continue
        name_candidates.append((min_stat_cy - it["cy"], clean_text))

    raw_name = min(name_candidates)[1] if name_candidates else "UNKNOWN"
    full_name, extra_tag = clean_display_name(raw_name)

    category_subtitle = None
    sub_candidates = [
        it["text"].strip() for it in profile_items
        if (min_stat_cy + 0.02) <= it["cy"] <= (min_stat_cy + 0.08) and it["x1"] < 0.45
    ]
    for sub in sub_candidates:
        if any(cat in sub.lower() for cat in ["creator", "artist", "model", "blogger", "actor", "coach", "figure", "trainer", "athlete"]):
            category_subtitle = sub
            break

    # 4. BIO, EMAIL & LOCATION
    all_image_lines = [it["text"] for it in filtered_items]
    bio_lines = [it["text"] for it in profile_items]
    full_bio_text = " \n ".join(bio_lines)

    # Search email across all lines in the image (including Contact bottom modal)
    email_candidates = _email_candidates_from_lines(all_image_lines, username=username)
    email = None
    if email_candidates:
        email_candidates.sort(key=lambda x: x[0], reverse=True)
        email = email_candidates[0][1]

    # LOCATION IS STRICTLY DETECTED FROM BIO ONLY (NEVER FROM HIGHLIGHTS OR REELS)
    location = GeoLocator.detect(full_bio_text)
    niche = NicheClassifier.detect(full_bio_text, category_subtitle, extra_tag)

    return {
        "email": email,
        "instagram_username": username,
        "full_name": full_name,
        "followers": stats["followers"],
        "following": stats["following"],
        "post_count": stats["post_count"],
        "location": location,
        "niche": niche,
        "bio": full_bio_text[:800],
    }


def _merge_missing(dst: dict, src: dict | None) -> dict:
    if not src:
        return dst
    for k in ("email", "instagram_username", "followers", "following", "post_count", "location", "niche"):
        if dst.get(k) is None:
            dst[k] = src.get(k)
    if (not dst.get("full_name")) or dst.get("full_name") == "UNKNOWN":
        dst["full_name"] = src.get("full_name") or dst.get("full_name")
    return dst


def multi_pass_20x_ocr(image_bytes: bytes, seq: int) -> dict:
    _tl.last_pass = None
    t0 = time.perf_counter()

    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            base_img = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid image file.") from exc

    full_w, full_h = base_img.size

    # Maintain full frame so bottom Contact drawers are readable
    if full_w > TARGET_OCR_WIDTH:
        ocr_scale = TARGET_OCR_WIDTH / float(full_w)
        target_ocr_height = int(full_h * ocr_scale)
        work_img = base_img.resize((TARGET_OCR_WIDTH, target_ocr_height), Image.Resampling.LANCZOS)
    else:
        ocr_scale = 1.0
        work_img = base_img.copy()

    gray_stat = ImageOps.grayscale(work_img)
    mean_brightness = np.array(gray_stat).mean()
    is_dark_mode = mean_brightness < 120.0
    log.info("[JOB #%d] [STEP 2/5] [PREPROCESS] Mode: %s (Brightness: %.1f) | Scaled %dx%d -> %dx%d",
             seq, "DarkMode" if is_dark_mode else "LightMode", mean_brightness, full_w, full_h, work_img.width, work_img.height)

    def run_ocr_pass(img_to_run: Image.Image) -> list | None:
        buf = io.BytesIO()
        img_to_run.save(buf, format="JPEG", quality=90)
        res, _ = get_engine()(buf.getvalue())
        if not res:
            return None
        unscaled = []
        for b, t, c in res:
            orig_box = [[p[0] / ocr_scale, p[1] / ocr_scale] for p in b]
            unscaled.append((orig_box, t, c))
        return unscaled

    # PASS 1: High Contrast
    log.info("[JOB #%d] [STEP 3/5] [OCR-20X] Executing Pass 1/3 (High Contrast Inversion)...", seq)
    pass1_img = ImageOps.invert(work_img) if is_dark_mode else work_img.copy()
    pass1_img = ImageOps.autocontrast(pass1_img, cutoff=1)
    res1 = run_ocr_pass(pass1_img)
    data1 = _extract_from_boxes(res1, full_w, full_h) if res1 else None

    if data1 and data1["email"] and data1["instagram_username"] and data1["followers"] is not None:
        log.info("[JOB #%d] [STEP 3/5] [OCR-20X] Pass 1 completed recovery in %.2fs", seq, time.perf_counter() - t0)
        _tl.last_pass = 1
        gc.collect()
        return data1

    # PASS 2: Edge Boost
    log.info("[JOB #%d] [STEP 3/5] [OCR-20X] Executing Pass 2/3 (Edge Boost Fallback)...", seq)
    pass2_img = ImageOps.invert(work_img) if is_dark_mode else work_img.copy()
    pass2_img = pass2_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2))
    pass2_img = ImageOps.autocontrast(pass2_img)
    res2 = run_ocr_pass(pass2_img)
    data2 = None
    if res2:
        data2 = _merge_missing(_extract_from_boxes(res2, full_w, full_h), data1)
        if data2["email"] and data2["instagram_username"]:
            log.info("[JOB #%d] [STEP 3/5] [OCR-20X] Pass 2 completed recovery in %.2fs", seq, time.perf_counter() - t0)
            _tl.last_pass = 2
            gc.collect()
            return data2

    # PASS 3: Deep Grayscale Fallback
    log.info("[JOB #%d] [STEP 3/5] [OCR-20X] Executing Pass 3/3 (Deep Grayscale Contrast)...", seq)
    pass3_img = ImageOps.grayscale(work_img)
    if is_dark_mode:
        pass3_img = ImageOps.invert(pass3_img)
    pass3_img = ImageEnhance.Contrast(pass3_img).enhance(2.0)
    res3 = run_ocr_pass(pass3_img)
    if res3:
        data3 = _merge_missing(_extract_from_boxes(res3, full_w, full_h), data2 or data1)
        _tl.last_pass = 3
        gc.collect()
        return data3

    _tl.last_pass = 3
    gc.collect()
    return data2 or data1 or {
        "email": None, "instagram_username": None, "full_name": "UNKNOWN",
        "followers": None, "following": None, "post_count": None,
        "location": None, "niche": None, "bio": None,
    }


def ocr_task(image_bytes: bytes, seq: int) -> tuple[dict, int | None]:
    data = multi_pass_20x_ocr(image_bytes, seq)
    return data, getattr(_tl, "last_pass", None)


# ----------------------------------------------------------------- Data Models
@dataclass
class Batch:
    chat_id: int
    started: float
    total: int = 0
    done: int = 0
    sent: int = 0
    new: int = 0
    dup: int = 0
    no_email: int = 0
    failed: int = 0
    lines: list = field(default_factory=list)
    failed_jobs: list = field(default_factory=list)
    running: dict = field(default_factory=dict)
    progress_msg: object = None
    last_added: float = 0.0
    finished: bool = False
    pending_buffer: list = field(default_factory=list)
    active_dispatched: int = 0


@dataclass
class Job:
    batch: Batch
    member: dict
    message: object
    file_id: str
    seq: int
    queued_at: float
    status: str = "failed"
    err: str = ""
    email: str | None = None
    extracted: dict = field(default_factory=dict)
    count: object = None
    t: dict = field(default_factory=lambda: {"dl": 0.0, "ocr": 0.0, "db": 0.0, "wait": 0.0, "total": 0.0, "pass": None})


BATCHES: dict[int, Batch] = {}


async def process_job(bot, job: Job) -> None:
    member, msg = job.member, job.message
    loop = asyncio.get_running_loop()

    # STEP 1: Download
    log.info("[JOB #%d] [STEP 1/5] [DOWNLOAD] Downloading photo from Telegram CDN...", job.seq)
    t = time.perf_counter()
    tg_file = await tg_call(bot.get_file, job.file_id)
    raw_bytes = bytes(await tg_call(tg_file.download_as_bytearray))
    job.t["dl"] = time.perf_counter() - t
    log.info("[JOB #%d] [STEP 1/5] [DOWNLOAD] Complete in %.2fs (%d KB)", job.seq, job.t["dl"], len(raw_bytes) // 1024)

    # STEP 2 & 3: 20x RapidOCR Multi-Pass
    t = time.perf_counter()
    extracted, used_pass = await loop.run_in_executor(OCR_POOL, ocr_task, raw_bytes, job.seq)
    job.t["ocr"] = time.perf_counter() - t
    job.t["pass"] = used_pass
    job.extracted = extracted
    email = extracted.get("email")
    job.email = email

    # STEP 4: Entity Logging
    log.info(
        "[JOB #%d] [STEP 4/5] [ENTITIES] Email: '%s' | Handle: @%s | Name: '%s' | Location: %s | Niche: %s",
        job.seq, email or "NOT_DETECTED", extracted.get("instagram_username") or "none",
        extracted.get("full_name"), extracted.get("location") or "none", extracted.get("niche") or "none"
    )

    # STEP 5: Supabase Persistence
    t = time.perf_counter()
    log.info("[JOB #%d] [STEP 5/5] [SUPABASE] Committing record to database...", job.seq)
    if not email:
        await run_db(lambda: db.table("creator_submissions").insert({
            "team_member_id": member["id"],
            "telegram_message_id": msg.message_id,
            "status": "email_not_found",
            "extracted_username": extracted.get("instagram_username"),
            "extracted_data": {k: v for k, v in extracted.items() if k != "email"},
        }).execute())
        job.status = "email_not_found"
        job.t["db"] = time.perf_counter() - t
        return

    rpc = await run_db(lambda: db.rpc("register_creator_lead", {
        "p_email": email,
        "p_username": extracted.get("instagram_username"),
        "p_full_name": extracted.get("full_name") or "UNKNOWN",
        "p_followers": extracted.get("followers"),
        "p_following": extracted.get("following"),
        "p_post_count": extracted.get("post_count"),
        "p_location": extracted.get("location"),
        "p_niche": extracted.get("niche"),
        "p_bio": extracted.get("bio"),
        "p_member_id": member["id"],
    }).execute())

    result = rpc.data[0] if isinstance(rpc.data, list) else rpc.data
    if not result or not result.get("creator_id"):
        raise RuntimeError("Database registration returned an empty payload.")

    creator_id = result["creator_id"]
    is_new = bool(result.get("is_new"))

    await run_db(lambda: db.table("creator_submissions").insert({
        "team_member_id": member["id"],
        "creator_id": creator_id,
        "telegram_message_id": msg.message_id,
        "status": "new" if is_new else "duplicate",
        "extracted_username": extracted.get("instagram_username"),
        "extracted_email": email,
    }).execute())

    try:
        job.count = await get_daily_count(str(member["id"]))
    except Exception:
        job.count = "?"

    job.status = "new" if is_new else "duplicate"
    job.t["db"] = time.perf_counter() - t
    log.info("[JOB #%d] [COMPLETED] Status: %s in %.2fs (OCR: %.2fs)", job.seq, job.status, time.perf_counter() - job.queued_at, job.t["ocr"])


def build_caption(job: Job) -> str:
    b, ex, t = job.batch, job.extracted, job.t
    status_header = {
        "new": "✅ [STATUS: NEW UNIQUE LEAD]",
        "duplicate": "⚠️ [STATUS: DUPLICATE LEAD]",
        "email_not_found": "❌ [STATUS: NO BUSINESS EMAIL DETECTED]",
        "failed": "🛑 [STATUS: FAILED TO PROCESS]",
    }[job.status]

    lines = [
        f"{status_header}  (#{job.seq})",
        f"Batch: {b.done}/{b.total} processed | Time: {t['total']:.1f}s",
        "----------------------------------------",
    ]
    if job.status == "failed":
        lines.append(f"Error: {job.err}")
    else:
        if job.email:
            lines.append(f"Email: {job.email}")
        lines += [
            f"Instagram: @{ex.get('instagram_username') or 'not detected'}",
            f"Name: {ex.get('full_name') or 'UNKNOWN'}",
            f"Posts: {_fmt(ex.get('post_count'))}",
            f"Followers: {_fmt(ex.get('followers'))}",
            f"Following: {_fmt(ex.get('following'))}",
            f"Location: {ex.get('location') or '—'}",
            f"Niche: {ex.get('niche') or '—'}",
        ]
        if job.status in ("new", "duplicate"):
            lines += ["", f"Daily Score: {job.count}/{DAILY_TARGET} unique leads"]

    pass_txt = f" (pass {t['pass']}/3)" if t.get("pass") else ""
    lines += [
        "----------------------------------------",
        f"Diagnostics: Wait {t['wait']:.1f}s | DL {t['dl']:.1f}s | OCR {t['ocr']:.1f}s{pass_txt} | DB {t['db']:.1f}s",
    ]
    return "\n".join(lines)[:1020]


def summary_line(job: Job) -> str:
    ex, t = job.extracted, job.t
    uname = ex.get("instagram_username") or "none"
    if job.status == "failed":
        return f"#{job.seq:02d} | 🛑 FAILED | Error: {job.err}"
    tag = {"new": "✅ NEW", "duplicate": "⚠️ DUP", "email_not_found": "❌ NO_EMAIL"}[job.status]
    return f"#{job.seq:02d} | {tag:<9} | {job.email or '—':<28} | @{uname:<18} | {_fmt(ex.get('followers'))} followers | {t['total']:.1f}s"


async def send_result(bot, job: Job, caption: str):
    try:
        await tg_call(job.message.reply_photo, photo=job.file_id, caption=caption)
    except Exception:
        await tg_call(bot.send_photo, chat_id=job.batch.chat_id, photo=job.file_id, caption=caption)


async def queue_worker(app: Application, wid: int):
    q: asyncio.Queue = app.bot_data["queue"]
    bot = app.bot
    while True:
        job: Job = await q.get()
        b = job.batch
        started = time.perf_counter()
        job.t["wait"] = time.time() - job.queued_at
        b.running[job.seq] = time.time()

        # Capture actual exception text instead of swallowing it
        success = False
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                await process_job(bot, job)
                success = True
                break
            except Exception as exc:
                last_exc = exc
                log.warning("[JOB #%d] Attempt %d failed: %s", job.seq, attempt + 1, exc, exc_info=True)
                await asyncio.sleep(1.0)

        if not success:
            job.status = "failed"
            job.err = short_err(last_exc) if last_exc else "Unknown failure"
            b.failed_jobs.append(job)

        job.t["total"] = time.perf_counter() - started
        b.running.pop(job.seq, None)
        b.done += 1
        if job.status == "new":
            b.new += 1
        elif job.status == "duplicate":
            b.dup += 1
        elif job.status == "email_not_found":
            b.no_email += 1
        else:
            b.failed += 1

        try:
            await send_result(bot, job, build_caption(job))
        except Exception:
            log.exception("Could not dispatch result message for #%d", job.seq)

        b.lines.append((job.seq, summary_line(job)))
        b.sent += 1
        q.task_done()

        # Refill queue: Keep active queue filled up to BATCH_CHUNK_SIZE
        b.active_dispatched = max(0, b.active_dispatched - 1)
        if b.pending_buffer and b.active_dispatched < BATCH_CHUNK_SIZE:
            next_job = b.pending_buffer.pop(0)
            b.active_dispatched += 1
            await q.put(next_job)


def progress_text(b: Batch, final: bool = False) -> str:
    now = time.time()
    elapsed = now - b.started
    total = max(b.total, 1)
    filled = int(10 * b.done / total)
    bar = "=" * filled + "-" * (10 - filled)
    running = sorted(b.running.items())
    running_txt = ", ".join(f"#{s} ({now - t0:.0f}s)" for s, t0 in running[:6]) or "None"
    waiting = max(0, b.total - b.done - len(b.running))

    timing = f"Elapsed: {fmt_secs(elapsed)}"
    if b.done:
        avg = elapsed / b.done
        timing += f" | Avg: {avg:.1f}s/img"
        if not final and b.done < b.total:
            timing += f" | ETA: ~{fmt_secs(avg * (b.total - b.done))}"

    title = "BATCH COMPLETED" if final else "PROCESSING BATCH"
    chunk_info = f" (Chunk window: {BATCH_CHUNK_SIZE})" if b.total > BATCH_CHUNK_SIZE else ""

    text = (
        f"[{title}]{chunk_info}\n"
        f"[{bar}] {b.done}/{b.total}\n\n"
        f"Verified Leads: {b.new} | Duplicate: {b.dup} | No Email: {b.no_email} | Failed: {b.failed}\n"
    )
    if not final:
        text += f"Active Slots: {running_txt}\nPending in Queue: {waiting}\n"
    text += f"{timing}\nWorkers Active: {OCR_WORKERS} Parallel slots"
    return text


async def batch_watcher(bot, member: dict, b: Batch):
    try:
        b.progress_msg = await tg_call(bot.send_message, b.chat_id, progress_text(b))
    except Exception:
        log.exception("Could not initiate progress notification")

    last_text = ""
    while True:
        await asyncio.sleep(PROGRESS_EVERY)
        if b.sent >= b.total and len(b.pending_buffer) == 0 and time.time() - b.last_added >= BATCH_IDLE_SECS:
            b.finished = True
            break
        text = progress_text(b)
        if b.progress_msg and text != last_text:
            last_text = text
            await safe_edit(b.progress_msg, text)

    if b.progress_msg:
        await safe_edit(b.progress_msg, progress_text(b, final=True))

    try:
        count = await get_daily_count(str(member["id"]))
    except Exception:
        count = "?"

    header = (
        f"FINAL BATCH AUDIT\n"
        f"Processed {b.total} screenshots in {fmt_secs(time.time() - b.started)}\n"
        f"New Unique: {b.new} | Duplicate: {b.dup} | No Email: {b.no_email} | Failed: {b.failed}\n"
        f"Your Total Unique Leads Today: {count}/{DAILY_TARGET}\n"
        f"{'=' * 40}\n"
    )
    body = "\n".join(line for _, line in sorted(b.lines))
    full_report = header + body

    for i in range(0, len(full_report), 3800):
        await tg_call(bot.send_message, b.chat_id, full_report[i:i + 3800])

    if b.failed_jobs:
        await tg_call(
            bot.send_message,
            b.chat_id,
            f"ATTENTION: {len(b.failed_jobs)} screenshot(s) could not be processed. Dispatched below for review:"
        )
        for failed_job in b.failed_jobs:
            caption = f"FAILED SCREENSHOT #{failed_job.seq}\nReason: {failed_job.err}\nPlease re-send this screenshot."
            try:
                await tg_call(bot.send_photo, chat_id=b.chat_id, photo=failed_job.file_id, caption=caption)
            except Exception:
                log.exception("Could not dispatch failed photo notification for #%d", failed_job.seq)


# ----------------------------------------------------------------- Telegram Handlers
def is_admin(uid: int) -> bool:
    return uid == ADMIN_ID


_MEMBER_CACHE: dict[int, tuple[float, dict]] = {}
_MEMBER_TTL = 60.0


async def get_member(uid: int):
    cached = _MEMBER_CACHE.get(uid)
    if cached and time.time() - cached[0] < _MEMBER_TTL:
        return cached[1]
    res = await run_db(lambda: db.table("team_members").select("*").eq("telegram_user_id", uid).eq("active", True).limit(1).execute())
    member = res.data[0] if res.data else None
    if member:
        _MEMBER_CACHE[uid] = (time.time(), member)
    else:
        _MEMBER_CACHE.pop(uid, None)
    return member


async def get_daily_count(member_id: str) -> int:
    today_iso = now_local().date().isoformat()
    res = await run_db(lambda: db.rpc("get_member_daily_progress", {"p_member_id": member_id, "p_day": today_iso}).execute())
    data = res.data[0] if isinstance(res.data, list) else res.data
    return int(data.get("email_count", 0)) if data else 0


def _fmt(val):
    return val if val is not None else "—"


def short_err(exc: Exception) -> str:
    text = re.sub(r"<[^>]+>", " ", str(exc))
    text = re.sub(r"\\\\r\\\\n|\\r\\n|\s+", " ", text).strip()
    return text[:160]


async def reply(update: Update, text: str):
    if update.effective_message:
        await update.effective_message.reply_text(text)


async def tg_call(fn, *args, retries: int = 5, **kwargs):
    for attempt in range(retries):
        try:
            return await fn(*args, **kwargs)
        except RetryAfter as exc:
            wait = exc.retry_after
            wait = wait.total_seconds() if hasattr(wait, "total_seconds") else float(wait)
            log.warning("Telegram flood limit reached, sleeping %.1fs", wait)
            await asyncio.sleep(wait + 0.5)
        except (TimedOut, NetworkError) as exc:
            log.warning("Telegram network issue (%s), retry %d/%d", exc, attempt + 1, retries)
            await asyncio.sleep(1 + attempt)
    return await fn(*args, **kwargs)


async def safe_edit(msg, text: str):
    try:
        await tg_call(msg.edit_text, text)
    except BadRequest as exc:
        if "not modified" not in str(exc).lower():
            log.debug("Message edit omitted: %s", exc)
    except Exception:
        pass


def fmt_secs(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 60:02d}:{s % 60:02d}"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return
    log.info("Received /start from @%s (ID: %d)", user.username, user.id)
    if is_admin(user.id):
        await run_db(lambda: db.table("team_members").upsert({
            "telegram_user_id": user.id, "telegram_username": user.username,
            "display_name": user.full_name or "Admin", "role": "admin", "active": True,
        }, on_conflict="telegram_user_id").execute())
        await reply(update, "Registered as admin.\nSend Instagram profile screenshots (one or many) to extract leads.")
    elif await get_member(user.id):
        await reply(update, "Welcome back. Send your creator screenshots to begin extraction.")
    else:
        await reply(update, "Access restricted. Please contact the administrator to be allowlisted.")


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not await get_member(user.id):
        return
    q: asyncio.Queue = context.application.bot_data["queue"]
    status_text = (
        "SYSTEM HEALTH MONITOR\n"
        f"Engine: Online\n"
        f"Parallel OCR Slots: {OCR_WORKERS} (x{OCR_THREADS} ONNX threads)\n"
        f"Throttled Chunk Window: {BATCH_CHUNK_SIZE} images\n"
        f"Database Concurrency: {DB_MAX_CONCURRENCY}\n"
        f"Active Processing Queue: {q.qsize()}"
    )
    await reply(update, status_text)


async def process_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user, msg = update.effective_user, update.effective_message
    if not user or not msg or (update.effective_chat and update.effective_chat.type != ChatType.PRIVATE):
        return

    log.info("--- [NEW SCREENSHOT ENQUEUED] from @%s (ID: %d) ---", user.username, user.id)
    try:
        member = await get_member(user.id)
    except Exception:
        log.exception("Member lookup failed")
        await msg.reply_text("Temporary database error, please resend this screenshot.")
        return
    if not member:
        log.warning("User %d is unauthorized.", user.id)
        await msg.reply_text("You are not registered. Ask the admin to add you.")
        return

    now = time.time()
    batch = BATCHES.get(user.id)
    if batch is None or batch.finished:
        batch = Batch(chat_id=msg.chat_id, started=now)
        BATCHES[user.id] = batch
        batch.watcher = asyncio.create_task(batch_watcher(context.bot, member, batch))

    batch.total += 1
    batch.last_added = now
    job = Job(batch=batch, member=member, message=msg, file_id=msg.photo[-1].file_id,
              seq=batch.total, queued_at=now)

    q: asyncio.Queue = context.application.bot_data["queue"]

    if batch.active_dispatched < BATCH_CHUNK_SIZE and q.qsize() < BATCH_CHUNK_SIZE:
        batch.active_dispatched += 1
        await q.put(job)
    else:
        batch.pending_buffer.append(job)


# ----------------------------------------------------------------- Health Server & Runner
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot service healthy")

    def log_message(self, format, *args):
        pass


def run_web_server():
    port = int(os.getenv("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()


async def post_init(app: Application):
    app.bot_data["queue"] = asyncio.Queue()
    _get_db_sem()
    await warm_up_engines()
    app.bot_data["workers"] = [
        asyncio.create_task(queue_worker(app, i)) for i in range(OCR_WORKERS + IO_EXTRA_WORKERS)
    ]
    log.info("System ready. Initialized %d worker threads (Chunk limit: %d)",
             OCR_WORKERS + IO_EXTRA_WORKERS, BATCH_CHUNK_SIZE)


async def post_shutdown(app: Application):
    for w in app.bot_data.get("workers", []):
        w.cancel()
    OCR_POOL.shutdown(wait=False, cancel_futures=True)


def main():
    if os.getenv("RUN_HEALTH_SERVER", "0") == "1":
        threading.Thread(target=run_web_server, daemon=True).start()
        log.info("Health check server running on port %s", os.getenv("PORT", 8080))

    req = HTTPXRequest(connection_pool_size=32, connect_timeout=30.0, read_timeout=60.0,
                       write_timeout=60.0, pool_timeout=30.0)
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(req)
        .concurrent_updates(True)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, process_photo))

    log.info("Starting production scraper engine with throttled chunk queue (Size: %d)", BATCH_CHUNK_SIZE)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()