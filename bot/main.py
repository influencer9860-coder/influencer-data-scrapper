from __future__ import annotations

import asyncio
import gc
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import logging
import os
import re
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from email_validator import EmailNotValidError, validate_email
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from rapidocr_onnxruntime import RapidOCR
from supabase import create_client
from telegram import Update
from telegram.constants import ChatType
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "DEBUG").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("creator-bot")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

DAILY_TARGET = int(os.getenv("DAILY_TARGET", "50"))
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "15"))
ADMIN_ID = int(os.getenv("AUTHORIZED_USER_ID", "0"))

try:
    TZ = ZoneInfo(os.getenv("TIMEZONE", "Asia/Kolkata"))
except Exception as exc:
    raise SystemExit(f"Invalid TIMEZONE: {exc}")

if not all([SUPABASE_URL, SUPABASE_KEY, BOT_TOKEN]) or ADMIN_ID <= 0:
    raise SystemExit("Missing required .env settings: SUPABASE_URL, SUPABASE_KEY, TELEGRAM_BOT_TOKEN, AUTHORIZED_USER_ID")

log.info("Connecting to Supabase at %s", SUPABASE_URL)
db = create_client(SUPABASE_URL, SUPABASE_KEY)

log.info("Loading 20x RapidOCR engine (ONNX Runtime)...")
engine = RapidOCR()
log.info("RapidOCR initialized successfully.")

# -----------------------------------------------------------------------------
# 20X TAXONOMY: LOCATIONS DATABASE
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
    "maharashtra": "Maharashtra, India", "karnataka": "Karnataka, India",
    "tamil nadu": "Tamil Nadu, India", "kerala": "Kerala, India",
    "gujarat": "Gujarat, India", "rajasthan": "Rajasthan, India",
    "punjab": "Punjab, India", "haryana": "Haryana, India",
    "uttar pradesh": "Uttar Pradesh, India", "madhya pradesh": "Madhya Pradesh, India",
    "bihar": "Bihar, India", "west bengal": "West Bengal, India",
    "odisha": "Odisha, India", "assam": "Assam, India",
    "himachal pradesh": "Himachal Pradesh, India", "uttarakhand": "Uttarakhand, India",
    "telangana": "Telangana, India", "andhra pradesh": "Andhra Pradesh, India",
    "goa": "Goa, India", "kashmir": "Jammu & Kashmir, India",
}

GLOBAL_LOCATIONS = {
    "dubai": "Dubai, UAE", "abu dhabi": "Abu Dhabi, UAE", "sharjah": "Sharjah, UAE",
    "london": "London, UK", "manchester": "Manchester, UK", "birmingham": "Birmingham, UK",
    "new york": "New York, USA", "nyc": "New York, USA", "los angeles": "Los Angeles, USA",
    "la": "Los Angeles, USA", "san francisco": "San Francisco, USA", "chicago": "Chicago, USA",
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
        r"\b(artist|art|painting|sketching|photographer|photography|filmmaker|cinematography|designer|design)\b"
    ),
    "Modeling & Editorial": (
        r"\b(model|modeling|editorial|runway|pageant|commercial\s*model|photoshoots?)\b"
    ),
    "Education & Books": (
        r"\b(educator|teacher|study|books?|bookstagram|reader|literature|coaching|motivation|mentor)\b"
    ),
}

VALID_TLD_RE = re.compile(
    r"\.(com|in|co\.in|org|net|io|ai|me|co|app|agency|media|info|biz|tech|store|xyz|live|tv|uk|us|ca|ae|de|fr|global)$",
    re.I
)

EMAIL_RE = re.compile(r"(?i)(?<![A-Z0-9._%+-])[A-Z0-9][A-Z0-9._%+-]*@[A-Z0-9.-]+\.[A-Z]{2,}(?![A-Z0-9.-])")
HANDLE_RE = re.compile(r"^[A-Za-z0-9._]{2,30}$")
COUNT_TOKEN_RE = re.compile(r"(?i)(?<![A-Za-z0-9])([0-9][0-9,.]*\s*[KMB]?)(?![A-Za-z])")
NUM_ONLY_RE = re.compile(r"^[0-9][0-9,.\s]*[KMBkmb]?$")

DOMAIN_TYPOS = {
    "gmai.com": "gmail.com",
    "gmial.com": "gmail.com",
    "gamil.com": "gmail.com",
    "gmaill.com": "gmail.com",
    "gmai1.com": "gmail.com",
    "yaho.com": "yahoo.com",
    "yahou.com": "yahoo.com",
    "outlok.com": "outlook.com",
    "hotmial.com": "hotmail.com",
    "icld.com": "icloud.com",
}

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


class EmailEngine:
    @staticmethod
    def clean(raw_candidate: str) -> str | None:
        if not raw_candidate or "@" not in raw_candidate:
            return None

        candidate = re.split(r"\.{2,}|…", raw_candidate)[0]
        candidate = re.sub(r"(?i)(\.(?:com|in|co\.in|org|net|io|ai|me|co|app|agency|media))(?:\s*(?:and\s*\d+\s*more|more).*)$", r"\1", candidate)
        candidate = candidate.replace("(at)", "@").replace("[at]", "@")
        candidate = re.sub(r"\s+", "", candidate).strip("<>()[]{}.,;:!?'\"-–—|•·")

        match = EMAIL_RE.search(candidate)
        if not match:
            return None

        email = match.group(0).lower().rstrip(".,;:-–—")
        if "@" not in email:
            return None

        local_part, domain = email.split("@", 1)

        local_part = re.sub(
            r"(?i)^.*?(?:collabs?|collaborat(?:e|ions?)|paid|pr|work|business|biz|enquir(?:y|ies)|queries|query|bookings?|email|mail|dm|contact|reach|connect|for)[-_:/|~]+\s*",
            "", local_part
        )
        local_part = re.sub(r"^[^a-z0-9]+", "", local_part)

        if domain in DOMAIN_TYPOS:
            domain = DOMAIN_TYPOS[domain]

        if not VALID_TLD_RE.search(domain):
            log.debug("  [REJECT INVALID TLD] '%s' is not an internet domain.", domain)
            return None

        if domain in {"gmail.com", "googlemail.com"} and len(local_part) < 6:
            return None
        if len(local_part) < 2:
            return None

        repaired_email = f"{local_part}@{domain}"
        try:
            return validate_email(repaired_email, check_deliverability=True).normalized.lower()
        except EmailNotValidError:
            if VALID_TLD_RE.search(domain) and re.fullmatch(r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$", repaired_email):
                return repaired_email
            return None
        except Exception:
            return repaired_email if "@" in repaired_email and "." in domain else None


class GeoLocator:
    @staticmethod
    def _scrub_token(token: str) -> str:
        t = re.sub(r"^[^a-zA-Z0-9]+", "", token)
        t = re.sub(r"[^a-zA-Z0-9]+$", "", t)
        if len(t) > 4 and t[0] in "PpQq09Oo" and t[1].isupper():
            t = t[1:]
        return t

    @classmethod
    def detect(cls, text: str) -> str | None:
        if not text:
            return None

        spaced_text = re.sub(r"[/|•·\n,–—~]+", " ", text)
        tokens = [cls._scrub_token(t) for t in spaced_text.split() if cls._scrub_token(t)]
        clean_sentence = " ".join(tokens)
        clean_lower = clean_sentence.lower()

        for city_key, canonical in INDIAN_CITIES.items():
            if re.search(r"(?i)(?:^|\s|[pP90oOqQ\W_])(" + re.escape(city_key) + r")(?:$|\s|[\W_]|l|I)", clean_lower):
                log.info("  [GEO ENGINE] Matched Indian City '%s' -> %s", city_key, canonical)
                return canonical

        for code, canonical in IATA_AIRPORT_CODES.items():
            if re.search(r"(?i)(?:^|\s)" + re.escape(code) + r"(?:$|\s)", clean_sentence):
                log.info("  [GEO ENGINE] Matched IATA Code '%s' -> %s", code, canonical)
                return canonical

        for glob_key, canonical in GLOBAL_LOCATIONS.items():
            if re.search(r"(?i)(?:^|\s|[pP90oOqQ\W_])(" + re.escape(glob_key) + r")(?:$|\s|[\W_])", clean_lower):
                log.info("  [GEO ENGINE] Matched Global Hub '%s' -> %s", glob_key, canonical)
                return canonical

        for state_key, canonical in INDIAN_STATES.items():
            if re.search(r"(?i)(?:^|\s)" + re.escape(state_key) + r"(?:$|\s)", clean_lower):
                log.info("  [GEO ENGINE] Matched Indian State '%s' -> %s", state_key, canonical)
                return canonical

        return None


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


def clean_display_name(raw_name: str) -> tuple[str, str | None]:
    clean = re.sub(r"[\u2600-\u27BF\U0001f300-\U0001f64f\U0001f680-\U0001f6ff\U0001f900-\U0001f9ff]+", " ", raw_name)
    clean = re.sub(r"('s)([A-Za-z])", r"\1 \2", clean)

    parts = re.split(r"\s*[|•·/–—\-]\s*", clean.strip())
    name = parts[0].strip()
    tag = parts[1].strip() if len(parts) > 1 and len(parts[1].strip()) >= 3 else None

    name = re.sub(r"\s+", " ", name).strip()
    return (name if len(name) >= 2 else "UNKNOWN", tag)


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
        if it["text"].lower() in {"follow", "following", "message", "edit profile", "contact", "share profile"} and it["cy"] > 0.25
    ]
    if action_buttons:
        button_y = min(action_buttons) + 0.02

    profile_items = [it for it in filtered_items if it["cy"] <= button_y]

    # --- 1. USERNAME ---
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

    # --- 2. STATS ---
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

    # --- 3. DISPLAY NAME ---
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

    # --- 4. BIO, EMAIL, LOCATION & NICHE ---
    all_lines = [it["text"] for it in profile_items]
    full_text = " \n ".join(all_lines)

    email_candidates = []
    email_indicator_pattern = re.compile(r"(?:✉|📧|📩|email|mail|contact|collab|queries|pr)", re.I)

    for line in all_lines:
        has_indicator = bool(email_indicator_pattern.search(line))
        norm = re.sub(r"@\s+", "@", line)
        norm = re.sub(r"([a-zA-Z0-9_.-]+)\s*\.\s*([a-zA-Z0-9_.-]+)", r"\1.\2", norm)

        for token in norm.split():
            if token.startswith("@"):
                continue

            e = EmailEngine.clean(token)
            if e:
                score = 100 if has_indicator else 50
                if any(provider in e for provider in ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"]):
                    score += 20
                email_candidates.append((score, e))
                log.info("  [EMAIL CANDIDATE] Found '%s' with score=%d (line: '%s')", e, score, line)

    email = None
    if email_candidates:
        email_candidates.sort(key=lambda x: x[0], reverse=True)
        email = email_candidates[0][1]
        log.info("  [EMAIL WINNER] Selected best email: %s", email)

    location = GeoLocator.detect(full_text)
    niche = NicheClassifier.detect(full_text, category_subtitle, extra_tag)

    return {
        "email": email,
        "instagram_username": username,
        "full_name": full_name,
        "followers": stats["followers"],
        "following": stats["following"],
        "post_count": stats["post_count"],
        "location": location,
        "niche": niche,
        "bio": full_text[:800],
    }


def multi_pass_20x_ocr(image_bytes: bytes) -> dict:
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            base_img = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid image file.") from exc

    full_w, full_h = base_img.size
    crop_frac = 0.62 if full_h / max(full_w, 1) > 1.5 else 1.0
    cropped = base_img.crop((0, 0, full_w, int(full_h * crop_frac)))

    # CLAMP RESOLUTION FOR RENDER 512MB RAM ENVIRONMENT
    # Mobile screenshots > 720px width exceed RAM when processed in full ONNX matrices.
    target_ocr_width = 720
    if cropped.width > target_ocr_width:
        ocr_scale = target_ocr_width / float(cropped.width)
        target_ocr_height = int(cropped.height * ocr_scale)
        work_img = cropped.resize((target_ocr_width, target_ocr_height), Image.Resampling.LANCZOS)
        log.info("Optimized input for Render RAM: %dx%d -> %dx%d (scale: %.3f)", full_w, full_h, target_ocr_width, target_ocr_height, ocr_scale)
    else:
        ocr_scale = 1.0
        work_img = cropped.copy()

    gray_stat = ImageOps.grayscale(work_img)
    mean_brightness = np.array(gray_stat).mean()
    is_dark_mode = mean_brightness < 120.0
    log.info("Screenshot brightness=%.1f (Dark Mode: %s)", mean_brightness, is_dark_mode)

    def run_ocr_pass(img_to_run: Image.Image) -> list | None:
        buf = io.BytesIO()
        img_to_run.save(buf, format="JPEG", quality=90)
        res, _ = engine(buf.getvalue())
        if not res:
            return None
        # Project coordinates back to original full screenshot scale
        unscaled_res = []
        for b, t, c in res:
            orig_box = [[p[0] / ocr_scale, p[1] / ocr_scale] for p in b]
            unscaled_res.append((orig_box, t, c))
        return unscaled_res

    # --- PASS 1: Auto-Inverted High Contrast Scan ---
    log.info("=== [PASS 1/3] Auto-Inverted High Contrast Scan ===")
    pass1_img = ImageOps.invert(work_img) if is_dark_mode else work_img.copy()
    pass1_img = ImageOps.autocontrast(pass1_img, cutoff=1)
    res1 = run_ocr_pass(pass1_img)
    data1 = _extract_from_boxes(res1, full_w, full_h) if res1 else None

    if data1 and data1["email"] and data1["instagram_username"] and data1["followers"] is not None:
        log.info("✅ Pass 1 extracted all primary fields successfully.")
        gc.collect()
        return data1

    # --- PASS 2: Sharpening & Enhanced Contrast (Memory Safe) ---
    log.info("=== [PASS 2/3] Sharpening & Edge Boost ===")
    pass2_img = work_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2))
    pass2_img = ImageOps.autocontrast(pass2_img)
    res2 = run_ocr_pass(pass2_img)
    if res2:
        data2 = _extract_from_boxes(res2, full_w, full_h)
        if data1:
            data2["email"] = data2["email"] or data1.get("email")
            data2["instagram_username"] = data2["instagram_username"] or data1.get("instagram_username")
            data2["followers"] = data2["followers"] or data1.get("followers")
            data2["post_count"] = data2["post_count"] or data1.get("post_count")
            data2["location"] = data2["location"] or data1.get("location")
            data2["niche"] = data2["niche"] or data1.get("niche")
        if data2["email"] and data2["instagram_username"]:
            log.info("✅ Pass 2 successfully completed recovery.")
            gc.collect()
            return data2
    else:
        data2 = None

    # --- PASS 3: Grayscale Contrast Fallback ---
    log.info("=== [PASS 3/3] Dynamic Grayscale Contrast Fallback ===")
    pass3_img = ImageOps.grayscale(work_img)
    pass3_img = ImageEnhance.Contrast(pass3_img).enhance(2.0)
    res3 = run_ocr_pass(pass3_img)
    if res3:
        data3 = _extract_from_boxes(res3, full_w, full_h)
        best = data2 or data1
        if best:
            data3["email"] = data3["email"] or best.get("email")
            data3["instagram_username"] = data3["instagram_username"] or best.get("instagram_username")
            data3["followers"] = data3["followers"] or best.get("followers")
            data3["post_count"] = data3["post_count"] or best.get("post_count")
            data3["location"] = data3["location"] or best.get("location")
            data3["niche"] = data3["niche"] or best.get("niche")
        gc.collect()
        return data3

    gc.collect()
    return data2 or data1 or {
        "email": None, "instagram_username": None, "full_name": "UNKNOWN",
        "followers": None, "following": None, "post_count": None,
        "location": None, "niche": None, "bio": None,
    }


# ----------------------------------------------------------------- Database
async def run_db(fn):
    return await asyncio.to_thread(fn)

def is_admin(uid: int) -> bool:
    return uid == ADMIN_ID

async def get_member(uid: int):
    res = await run_db(lambda: db.table("team_members").select("*").eq("telegram_user_id", uid).eq("active", True).limit(1).execute())
    return res.data[0] if res.data else None

async def get_daily_count(member_id: str) -> int:
    today_iso = now_local().date().isoformat()
    res = await run_db(lambda: db.rpc("get_member_daily_progress", {"p_member_id": member_id, "p_day": today_iso}).execute())
    data = res.data[0] if isinstance(res.data, list) else res.data
    return int(data.get("email_count", 0)) if data else 0

def _fmt(val):
    return val if val is not None else "—"

async def reply(update: Update, text: str):
    if update.effective_message:
        await update.effective_message.reply_text(text)


# ----------------------------------------------------------------- Telegram Handlers
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
        await reply(update, "Registered as admin.\nSend an Instagram profile screenshot to extract leads.")
    elif await get_member(user.id):
        await reply(update, "You're registered. Send an Instagram profile screenshot.")
    else:
        await reply(update, "You are not on the allowlist. Ask the admin to add you.")


async def process_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user, msg = update.effective_user, update.effective_message
    if not user or not msg or (update.effective_chat and update.effective_chat.type != ChatType.PRIVATE):
        return

    log.info("--- [NEW SCREENSHOT] from @%s (ID: %d) ---", user.username, user.id)
    member = await get_member(user.id)
    if not member:
        log.warning("User %d is unauthorized.", user.id)
        await msg.reply_text("You are not registered. Ask the admin to add you.")
        return

    photo = msg.photo[-1]
    status = await msg.reply_text("⏳ [1/3] Downloading screenshot...")
    try:
        tg_file = await context.bot.get_file(photo.file_id)
        raw_bytes = bytes(await tg_file.download_as_bytearray())

        await status.edit_text("🔍 [2/3] Analyzing with 20x Deep Engine...")
        extracted = await asyncio.to_thread(multi_pass_20x_ocr, raw_bytes)
        email = extracted.get("email")

        await status.edit_text("💾 [3/3] Registering lead in Supabase...")
        if not email:
            log.info("No email found in screenshot. Storing review record.")
            await run_db(lambda: db.table("creator_submissions").insert({
                "team_member_id": member["id"],
                "telegram_message_id": msg.message_id,
                "status": "email_not_found",
                "extracted_username": extracted.get("instagram_username"),
                "extracted_data": {k: v for k, v in extracted.items() if k != "email"},
            }).execute())

            await status.edit_text(
                "❌ No public business email detected.\n\n"
                f'Instagram: @{extracted.get("instagram_username") or "not detected"}\n'
                f'Name: {extracted.get("full_name") or "UNKNOWN"}\n'
                f'Followers: {_fmt(extracted.get("followers"))} | Posts: {_fmt(extracted.get("post_count"))}'
            )
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

        count = await get_daily_count(str(member["id"]))
        label = "✅ NEW UNIQUE EMAIL" if is_new else "⚠️ DUPLICATE EMAIL — NO EXTRA CREDIT"

        await status.edit_text(
            f"{label}\n\n"
            f"Email: {email}\n"
            f'Instagram: @{extracted.get("instagram_username") or "not detected"}\n'
            f'Name: {extracted.get("full_name") or "UNKNOWN"}\n'
            f'Posts: {_fmt(extracted.get("post_count"))}\n'
            f'Followers: {_fmt(extracted.get("followers"))}\n'
            f'Following: {_fmt(extracted.get("following"))}\n'
            f'Location: {extracted.get("location") or "—"}\n'
            f'Niche: {extracted.get("niche") or "—"}\n\n'
            f"Your unique emails today: {count}/{DAILY_TARGET}"
        )
    except Exception as exc:
        log.exception("Processing failed")
        await status.edit_text(f"❌ Failed to process screenshot: {exc}")


# ----------------------------------------------------------------- Built-in Health Server
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running!")

    def log_message(self, format, *args):
        pass


def run_web_server():
    port = int(os.getenv("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()


def main():
    threading.Thread(target=run_web_server, daemon=True).start()
    log.info("Render health check server started on port %s", os.getenv("PORT", 8080))

    req = HTTPXRequest(connection_pool_size=16, connect_timeout=30.0, read_timeout=60.0)
    app = Application.builder().token(BOT_TOKEN).request(req).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, process_photo))

    log.info("Starting Telegram bot with 20x Deep RapidOCR engine")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()