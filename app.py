"""
Citizen Grievance web app (Flask).

How this file is organized (good reading order for beginners):
  1. Imports — bring in Flask tools and our database helpers.
  2. Settings — demo passwords, department lists, keyword sets.
  3. bootstrap_app() — runs once at import: create DB tables.
  4. Small helper functions — reused checks (who is logged in?, normalize URLs, etc.).
  5. inject_layout_context — fills sidebar / top bar on every HTML page.
  6. Route functions — each @app.route("/some-path") handles one URL.

Three visitor types:
  • Citizen — uses "/" (submit) and "/track" without logging in.
    • Department staff — "/auth" then "/dashboard/<department>".
    • Admin — "/auth" then "/admin", analytics, status updates for all departments.
"""

import difflib
import os
import random
import re
import time
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import joblib
import requests
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from sklearn.metrics.pairwise import cosine_similarity
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    VideoFileClip = None
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from database.db import (
    CATEGORY_TO_DEPARTMENT,
    CONTACT_AREAS,
    STATUS_FLOW,
    count_user_complaints,
    create_user_account,
    fetch_department_contact,
    fetch_complaints,
    fetch_open_complaints_same_area_category,
    fetch_user_complaints,
    get_complaint_notification_payload,
    get_analytics_data,
    get_complaint_by_uid,
    get_feedback_by_complaint_uid,
    get_latest_user_location,
    get_user_by_id,
    get_user_by_login,
    init_db,
    mark_user_verified,
    record_complaint_feedback,
    register_complaint,
    set_resolution_email_sent_at,
    set_complaint_status_by_id,
    set_complaints_high_priority,
    update_user_otp,
)

# =============================================================================
# SETTINGS (demo values — replace for a real deployment)
# =============================================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = "citizen-grievance-demo-secret"
ADMIN_PASSWORD = "admin"

UPLOAD_FOLDER = os.path.join("static", "uploads")
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
MAX_UPLOAD_FILES = 2
MAX_VIDEO_DURATION_SECONDS = 60
MAX_VIDEO_SIZE_MB = 25
MAX_REQUEST_SIZE_MB = 60
MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1024 * 1024
MAX_REQUEST_BYTES = MAX_REQUEST_SIZE_MB * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing pre-trained model file: {MODEL_PATH}")
if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"Missing pre-trained vectorizer file: {VECTORIZER_PATH}")

# Load pre-trained ML artifacts once at startup (no runtime training).
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def load_env_file(path: str) -> None:
    """Load simple KEY=VALUE pairs from a local .env file without extra deps."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"").strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        # OTP email can still work if vars are provided by the host environment.
        pass


load_env_file(os.path.join(BASE_DIR, ".env"))

BREVO_API_KEY = os.getenv("BREVO_API_KEY")
BREVO_SENDER_EMAIL = os.getenv("BREVO_SENDER_EMAIL", "your_verified_email@gmail.com")
SMTP_FROM = os.getenv("SMTP_FROM")
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = os.getenv("SMTP_PORT", "587")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true")

# Fake department accounts: username → password + internal department key
DEPARTMENT_USERS = {
    "water_user": {"password": "123", "department": "water"},
    "electricity_user": {"password": "123", "department": "electricity"},
    "roads_user": {"password": "123", "department": "roads"},
    "healthcare_user": {"password": "123", "department": "healthcare"},
    "garbage_user": {"password": "123", "department": "garbage"},
    "fire_user": {"password": "123", "department": "fire"},
}

# URL/session slug → category label stored on each complaint
DEPARTMENT_TO_CATEGORY = {
    "water": "Water",
    "electricity": "Electricity",
    "roads": "Roads",
    "healthcare": "Healthcare",
    "garbage": "Garbage",
    "fire": "Fire",
}

HIGH_PRIORITY_KEYWORDS = {"urgent", "emergency", "danger"}
MEDIUM_PRIORITY_KEYWORDS = {"soon", "delay"}
MODEL_LABEL_TO_CATEGORY = {
    "road": "Roads",
    "roads": "Roads",
    "water": "Water",
    "electricity": "Electricity",
    "garbage": "Garbage",
    "health": "Healthcare",
    "healthcare": "Healthcare",
    "fire": "Fire",
}

CATEGORY_KEYWORDS = {
    "Electricity": {
        "electricity",
        "power",
        "current",
        "bijli",
        "voltage",
        "outage",
        "blackout",
    },
    "Water": {
        "water",
        "pani",
        "paani",
        "jal",
        "tap",
        "pipeline",
        "supply",
        "leak",
        "leakage",
    },
    "Roads": {
        "road",
        "roads",
        "sadak",
        "pothole",
        "gaddha",
        "gadda",
        "footpath",
    },
    "Garbage": {
        "garbage",
        "kuda",
        "kooda",
        "kachra",
        "safai",
        "waste",
        "trash",
        "sanitation",
    },
    "Healthcare": {
        "health",
        "healthcare",
        "hospital",
        "clinic",
        "doctor",
        "medical",
        "ambulance",
    },
    "Fire": {
        "fire",
        "agni",
        "fire station",
        "fire brigade",
        "firestation",
    },
}

TRACK_STATUS_OPTIONS = ["Pending", "Approved", "In Progress", "Resolved", "Rejected"]
VIDEO_MIME_TYPES = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
}

CHATBOT_AREAS = CONTACT_AREAS
CHATBOT_MAX_MESSAGE_LEN = 400
CHATBOT_RATE_LIMIT = 8
CHATBOT_RATE_WINDOW_SECONDS = 60

REVERSE_GEOCODE_URL = "https://nominatim.openstreetmap.org/reverse"
REVERSE_GEOCODE_TIMEOUT_SECONDS = 8
LOCATION_LABEL_MAX_LEN = 120
IP_GEO_URL = "https://ipapi.co/json/"
IP_GEO_TIMEOUT_SECONDS = 6

DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097f]")
HINDI_PHRASE_REPLACEMENTS = [
    ("पानी की समस्या", "water problem"),
    ("बिजली नहीं आ रही", "electricity issue"),
    ("सड़क टूट गई", "road broken"),
    ("सड़क टूट गयी", "road broken"),
    ("सड़क टूट गई है", "road broken"),
    ("सड़क टूट गयी है", "road broken"),
    ("पानी लीक", "water leak"),
    ("जल लीक", "water leak"),
]

TOKEN_REPLACEMENTS = {
    "पानी": "water",
    "जल": "water",
    "सड़क": "road",
    "सडक़": "road",
    "रोड": "road",
    "मार्ग": "road",
    "बिजली": "electricity",
    "कचरा": "garbage",
    "कूड़ा": "garbage",
    "कूड़ा": "garbage",
    "सफाई": "garbage",
    "नाली": "sewage",
    "सीवर": "sewage",
    "सीवरेज": "sewage",
    "आग": "fire",
    "अग्नि": "fire",
    "फायर": "fire",
    "स्वास्थ्य": "healthcare",
    "हॉस्पिटल": "hospital",
    "अस्पताल": "hospital",
    "डॉक्टर": "doctor",
    "समस्या": "problem",
    "मुद्दा": "issue",
    "शिकायत": "complaint",
    "लीक": "leak",
    "लीकेज": "leakage",
    "टूट": "broken",
    "टूटा": "broken",
    "टूटी": "broken",
    "खराब": "damaged",
    "गड्ढा": "pothole",
    "गड्डा": "pothole",
    "प्रॉब्लम": "problem",
    "समाधान": "resolved",
    "आपातकाल": "emergency",
    "तुरंत": "urgent",
    "खतरा": "danger",
    "जल्दी": "soon",
    "pani": "water",
    "jal": "water",
    "sadak": "road",
    "road": "road",
    "bijli": "electricity",
    "current": "electricity",
    "power": "electricity",
    "kachra": "garbage",
    "kuda": "garbage",
    "kooda": "garbage",
    "safai": "garbage",
    "nali": "sewage",
    "naali": "sewage",
    "sewer": "sewage",
    "sewage": "sewage",
    "drainage": "sewage",
    "drain": "sewage",
    "gutter": "sewage",
    "aag": "fire",
    "fire": "fire",
    "hospital": "hospital",
    "doctor": "doctor",
    "clinic": "clinic",
    "health": "healthcare",
    "healthcare": "healthcare",
    "issue": "issue",
    "problem": "problem",
    "complaint": "complaint",
    "leak": "leak",
    "leakage": "leakage",
    "broken": "broken",
    "damage": "damage",
    "damaged": "damage",
    "pothole": "pothole",
    "gaddha": "pothole",
    "gadda": "pothole",
}

ENGLISH_NORMALIZATION_RULES = [
    (r"\bwater\s+leak(ing|age)?\b", "water leakage"),
    (r"\bleak(ing|age)?\s+water\b", "water leakage"),
    (r"\broad\s+(broken|damage|damaged|tut|toot|tooti|tuti)\b", "road damage"),
    (r"\broad\s+pothole(s)?\b", "road pothole"),
    (r"\belectricity\s+(problem|issue|cut|outage|failure)\b", "electricity issue"),
    (r"\bpower\s+(cut|outage|failure|problem|issue)\b", "electricity issue"),
    (r"\bgarbage\s+(problem|issue|collection|dump)\b", "garbage issue"),
]

SEWAGE_HINT_TERMS = {"sewage", "drainage", "drain", "gutter"}

CHATBOT_DEPARTMENT_KEYWORDS = {
    "Water Department": [
        "water",
        "pani",
        "paani",
        "jal",
        "water supply",
        "tap",
        "pipeline",
    ],
    "Electricity Department": [
        "electricity",
        "bijli",
        "power",
        "current",
        "voltage",
        "light bill",
    ],
    "Road/Municipal Department": [
        "road",
        "roads",
        "sadak",
        "pothole",
        "gaddha",
        "municipal",
        "footpath",
    ],
    "Garbage Department": [
        "sanitation",
        "garbage",
        "kuda",
        "safai",
        "waste",
        "trash",
    ],
    "Healthcare Department": [
        "health",
        "healthcare",
        "hospital",
        "clinic",
        "medical",
        "doctor",
        "ambulance",
    ],
    "Fire Department": [
        "fire",
        "agni",
        "fire brigade",
        "firestation",
        "fire station",
    ],
}

TRANSLATIONS = {
    "en": {
        "account": "Account",
        "admin_login": "Admin Login",
        "admin_password": "Admin Password",
        "analytics": "Analytics",
        "choose_login_type": "Choose a login type to continue.",
        "citizen_access": "Citizen Access",
        "category": "Category",
        "complaint_text": "Complaint Text",
        "in_progress": "In Progress",
        "confirm_password": "Confirm Password",
        "create_account": "Create Account",
        "dashboard": "Dashboard",
        "department": "Department",
        "department_login": "Department Login",
        "email": "Email",
        "login": "Login",
        "login_as_admin": "Login as Admin",
        "login_as_department": "Login as Department",
        "login_create_account": "Login or create your account to submit and track grievances.",
        "location": "Location",
        "logout": "Logout",
        "main": "Main",
        "password": "Password",
        "priority": "Priority",
        "register_complaint": "Register Complaint",
        "search": "Search",
        "search_placeholder": "Search by ID, keyword, location...",
        "sign_in": "Sign In",
        "sign_up": "Sign Up",
        "staff": "Staff",
        "staff_access": "Staff Access",
        "submit": "Submit Complaint",
        "submitted_on": "Submitted on",
        "submitted": "Submitted",
        "track": "Track Complaint",
        "resolved": "Resolved",
        "track_your_complaint": "Track Your Complaint",
        "username": "Username",
        "username_or_email": "Username or Email",
        "verify": "Verify",
        "verify_email": "Verify Email",
        "otp_code": "OTP Code",
        "otp_sent_to": "We sent a 6-digit OTP to",
        "resend_otp": "Resend OTP",
    },
    "hi": {
        "account": "खाता",
        "admin_login": "एडमिन लॉगिन",
        "admin_password": "एडमिन पासवर्ड",
        "analytics": "विश्लेषण",
        "choose_login_type": "जारी रखने के लिए लॉगिन प्रकार चुनें।",
        "citizen_access": "नागरिक एक्सेस",
        "category": "श्रेणी",
        "complaint_text": "शिकायत विवरण",
        "in_progress": "प्रक्रिया में",
        "confirm_password": "पासवर्ड की पुष्टि",
        "create_account": "खाता बनाएं",
        "dashboard": "डैशबोर्ड",
        "department": "विभाग",
        "department_login": "विभाग लॉगिन",
        "email": "ईमेल",
        "login": "लॉगिन",
        "login_as_admin": "एडमिन के रूप में लॉगिन",
        "login_as_department": "विभाग के रूप में लॉगिन",
        "login_create_account": "शिकायत दर्ज करने और ट्रैक करने के लिए लॉगिन करें या खाता बनाएं।",
        "location": "स्थान",
        "logout": "लॉगआउट",
        "main": "मुख्य",
        "password": "पासवर्ड",
        "priority": "प्राथमिकता",
        "register_complaint": "शिकायत दर्ज करें",
        "search": "खोजें",
        "search_placeholder": "आईडी, कीवर्ड, स्थान से खोजें...",
        "sign_in": "साइन इन",
        "sign_up": "साइन अप",
        "staff": "स्टाफ",
        "staff_access": "स्टाफ एक्सेस",
        "submit": "शिकायत दर्ज करें",
        "submitted_on": "दर्ज की गई तारीख",
        "submitted": "दर्ज",
        "track": "शिकायत ट्रैक",
        "resolved": "समाधान",
        "track_your_complaint": "अपनी शिकायत ट्रैक करें",
        "username": "यूजरनेम",
        "username_or_email": "यूजरनेम या ईमेल",
        "verify": "सत्यापित करें",
        "verify_email": "ईमेल सत्यापन",
        "otp_code": "ओटीपी कोड",
        "otp_sent_to": "हमने 6-अंकों का ओटीपी भेजा है",
        "resend_otp": "ओटीपी फिर भेजें",
    },
}


def get_text(key: str) -> str:
    """Translate UI labels using the language stored in session."""
    lang = session.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


def bootstrap_app() -> None:
    """Prepare SQLite schema before any HTTP requests."""
    init_db()


bootstrap_app()


# =============================================================================
# HELPERS — session & department checks
# =============================================================================


def normalize_department_slug(department: str) -> str:
    """Turn 'Water_Dept' / 'water-dept' style input into one stable slug like 'water-dept'."""
    return "-".join(department.strip().lower().replace("_", " ").replace("-", " ").split())


def current_department_from_session() -> Optional[str]:
    """Department slug if the browser session belongs to a valid department user; else None."""
    department = session.get("department")
    if not isinstance(department, str):
        return None
    slug = normalize_department_slug(department)
    if slug not in DEPARTMENT_TO_CATEGORY:
        return None
    return slug


def is_admin_logged_in() -> bool:
    """True only after a successful /auth login as admin."""
    return session.get("is_admin") is True


def is_user_logged_in() -> bool:
    """True when a citizen user account is logged in."""
    return session.get("user_id") is not None


def department_panel_title(department_slug: str) -> str:
    """Human title shown on a department dashboard."""
    category_name = DEPARTMENT_TO_CATEGORY.get(department_slug, "Department")
    return f"{category_name} Department Panel"


def priority_from_text(complaint_text: str) -> str:
    """Very simple rule: scan words for configured keywords → High / Medium / Low."""
    cleaned_text = preprocess_complaint_text(complaint_text)
    tokens = set(cleaned_text.split()) if cleaned_text else set()

    if tokens.intersection(HIGH_PRIORITY_KEYWORDS):
        return "High"
    if tokens.intersection(MEDIUM_PRIORITY_KEYWORDS):
        return "Medium"
    return "Low"


def normalize_location(location: str) -> str:
    """Normalize user-provided locations for repeat detection."""
    tokens = re.findall(r"[a-z0-9]+", location.lower())
    return " ".join(tokens).strip()


def preprocess_complaint_text(text: str) -> str:
    """Normalize Hindi/Hinglish/English text to assist ML + keyword routing."""
    raw = str(text or "").strip().lower()
    if not raw:
        return ""

    cleaned = raw.replace("।", " ")
    cleaned = re.sub(r"[\"'`]+", "", cleaned)
    cleaned = re.sub(r"[\(\)\[\]{}<>]", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\u0900-\u097f\s]", " ", cleaned)
    cleaned = " ".join(cleaned.split())

    is_hindi = bool(DEVANAGARI_REGEX.search(cleaned))
    if is_hindi:
        for phrase, replacement in HINDI_PHRASE_REPLACEMENTS:
            cleaned = cleaned.replace(phrase, replacement)

    for pattern, replacement in ENGLISH_NORMALIZATION_RULES:
        cleaned = re.sub(pattern, replacement, cleaned)

    tokens = cleaned.split()
    mapped_tokens = [TOKEN_REPLACEMENTS.get(token, token) for token in tokens]
    cleaned = " ".join(mapped_tokens)

    cleaned = re.sub(r"[\u0900-\u097f]", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = " ".join(cleaned.split())

    for pattern, replacement in ENGLISH_NORMALIZATION_RULES:
        cleaned = re.sub(pattern, replacement, cleaned)

    tokens = cleaned.split()
    if any(term in tokens for term in SEWAGE_HINT_TERMS) and "garbage" not in tokens:
        cleaned = f"{cleaned} garbage".strip()

    return cleaned


def detect_category_from_text(text: str) -> Optional[str]:
    """Return the category whose keyword appears first in the text."""
    normalized = preprocess_complaint_text(text)
    if not normalized:
        return None

    best_category = None
    best_index = None
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if " " in keyword:
                match = re.search(re.escape(keyword), normalized)
            else:
                match = re.search(r"\b" + re.escape(keyword) + r"\b", normalized)
            if not match:
                continue
            index = match.start()
            if best_index is None or index < best_index:
                best_index = index
                best_category = category

    return best_category


def parse_coordinate(value: Optional[str], min_value: float, max_value: float) -> Optional[float]:
    """Parse and validate a coordinate value."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        coord = float(raw)
    except ValueError:
        return None
    if coord < min_value or coord > max_value:
        return None
    return round(coord, 6)


def build_location_label(area: Optional[str], city: Optional[str], fallback: str) -> str:
    """Build a short display label from area + city, or fallback text."""
    area_value = str(area or "").strip()
    city_value = str(city or "").strip()
    if area_value and city_value and area_value.lower() not in city_value.lower():
        label = f"{area_value}, {city_value}"
    elif area_value:
        label = area_value
    else:
        label = fallback
    return label[:LOCATION_LABEL_MAX_LEN]


def similarity_scores(new_text: str, candidates: List[str]) -> List[float]:
    """Return similarity scores (TF-IDF cosine) with a simple fallback."""
    if not candidates:
        return []
    try:
        new_vec = vectorizer.transform([new_text])
        candidate_vecs = vectorizer.transform(candidates)
        return cosine_similarity(new_vec, candidate_vecs)[0].tolist()
    except Exception:
        new_tokens = set(re.findall(r"[a-z0-9]+", new_text.lower()))
        scores = []
        for text in candidates:
            tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
            union = new_tokens | tokens
            scores.append(len(new_tokens & tokens) / max(1, len(union)))
        return scores


def auto_escalate_priority(
    complaint_id: int,
    location_key: str,
    category: str,
    complaint_text: str,
    similarity_threshold: float = 0.35,
    min_count: int = 3,
) -> bool:
    """Escalate priority when repeat complaints appear in the same area."""
    related = fetch_open_complaints_same_area_category(
        location_key=location_key,
        category=category,
        exclude_id=complaint_id,
    )
    candidate_texts = [row["text"] for row in related]
    scores = similarity_scores(complaint_text, candidate_texts)
    matched_ids = [
        row["complaint_id"]
        for row, score in zip(related, scores)
        if score >= similarity_threshold
    ]

    total_related = len(matched_ids) + 1
    if total_related >= min_count:
        group_key = f"{location_key}:{category}"
        set_complaints_high_priority(
            complaint_ids=matched_ids + [complaint_id],
            escalation_reason="Repeated complaints in same area",
            escalation_group=group_key,
            escalation_count=total_related,
        )
        return True
    return False


def classify_complaint(text: str) -> str:
    """Classify grievance text using pre-trained model + vectorizer."""
    explicit_category = detect_category_from_text(text)
    if explicit_category:
        return explicit_category

    normalized = preprocess_complaint_text(text)
    payload = normalized if normalized else str(text or "")
    X = vectorizer.transform([payload])
    prediction = model.predict(X)[0]
    prediction_label = str(prediction).strip().lower()
    return MODEL_LABEL_TO_CATEGORY.get(prediction_label, "Roads")


def file_extension(filename: str) -> str:
    """Return lowercase file extension (without dot), or empty string."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def allowed_file(filename: str) -> bool:
    """Allow only configured image/video file extensions."""
    return file_extension(filename) in ALLOWED_EXTENSIONS


def normalize_chat_text(text: str) -> str:
    """Normalize user text for simple keyword matching."""
    return preprocess_complaint_text(text)


def detect_department_from_message(message: str) -> Optional[str]:
    """Infer department from keywords in the message."""
    normalized = normalize_chat_text(message)
    tokens = normalized.split()

    for department, keywords in CHATBOT_DEPARTMENT_KEYWORDS.items():
        for keyword in keywords:
            keyword_norm = normalize_chat_text(keyword)
            if not keyword_norm:
                continue
            if " " in keyword_norm:
                if keyword_norm in normalized:
                    return department
            else:
                if keyword_norm in tokens:
                    return department

    dept_labels = [normalize_chat_text(name) for name in CHATBOT_DEPARTMENT_KEYWORDS]
    fuzzy_match = difflib.get_close_matches(normalized, dept_labels, n=1, cutoff=0.72)
    if fuzzy_match:
        idx = dept_labels.index(fuzzy_match[0])
        return list(CHATBOT_DEPARTMENT_KEYWORDS.keys())[idx]
    return None


def detect_area_from_message(message: str) -> Optional[str]:
    """Detect area from known area list."""
    normalized = normalize_chat_text(message)
    tokens = normalized.split()
    collapsed = "".join(tokens)

    for area in CHATBOT_AREAS:
        area_norm = normalize_chat_text(area)
        area_tokens = area_norm.split()
        if area_tokens and all(token in tokens for token in area_tokens):
            return area
        area_collapsed = "".join(area_tokens)
        if area_collapsed and area_collapsed in collapsed:
            return area

    area_token_map = []
    for area in CHATBOT_AREAS:
        area_norm = normalize_chat_text(area)
        for token in area_norm.split():
            if len(token) >= 4:
                area_token_map.append((token, area))

    area_tokens_only = [token for token, _ in area_token_map]
    for token in tokens:
        if len(token) < 4:
            continue
        match = difflib.get_close_matches(token, area_tokens_only, n=1, cutoff=0.78)
        if match:
            matched_token = match[0]
            for token_value, area in area_token_map:
                if token_value == matched_token:
                    return area

    area_lookup = {normalize_chat_text(area): area for area in CHATBOT_AREAS}
    close = difflib.get_close_matches(normalized, list(area_lookup.keys()), n=1, cutoff=0.7)
    if close:
        return area_lookup[close[0]]
    return None


def resolve_area_from_location(location: Optional[str]) -> Optional[str]:
    """Map a stored location to a known area name."""
    if not location:
        return None
    return detect_area_from_message(location)


def is_chatbot_rate_limited() -> bool:
    """Simple session-based rate limiting for chatbot requests."""
    now = time.time()
    hits = session.get("chatbot_hits", [])
    recent = [ts for ts in hits if now - float(ts) < CHATBOT_RATE_WINDOW_SECONDS]
    if len(recent) >= CHATBOT_RATE_LIMIT:
        session["chatbot_hits"] = recent
        return True
    recent.append(now)
    session["chatbot_hits"] = recent
    return False


def cleanup_uploaded_files(filenames) -> None:
    """Delete uploaded files best-effort when validation fails."""
    for filename in filenames:
        if not filename:
            continue
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            # Cleanup errors should not block complaint flow handling.
            pass


def dashboard_summary_counts(complaints) -> Dict[str, int]:
    """Numbers for the colored summary cards on admin / department dashboards."""
    return {
        "total": len(complaints),
        "pending": sum(1 for row in complaints if row["status"] == "Pending"),
        "in_progress": sum(1 for row in complaints if row["status"] == "In Progress"),
        "resolved": sum(1 for row in complaints if row["status"] == "Resolved"),
        "high_priority": sum(1 for row in complaints if row["priority"] == "High"),
    }


def build_escalation_alerts(complaints: List[Dict[str, Any]]) -> List[str]:
    """Return a short list of escalated locations for dashboard alerts."""
    locations = []
    for row in complaints:
        if row.get("is_escalated") and row.get("location"):
            locations.append(str(row["location"]))
    if not locations:
        return []
    deduped = []
    for item in locations:
        if item not in deduped:
            deduped.append(item)
    return deduped[:3]


def complaint_progress_details(status: str) -> Tuple[int, str]:
    """Progress % + short label for citizen complaint timeline cards."""
    status_map = {
        "Pending": (20, "Complaint received"),
        "Approved": (45, "Complaint approved"),
        "In Progress": (72, "Department is working on it"),
        "Resolved": (100, "Issue resolved"),
        "Rejected": (100, "Complaint rejected"),
    }
    return status_map.get(str(status or "").strip(), (20, "Complaint received"))


def build_proof_assets(media_value: str) -> List[Dict[str, str]]:
    """Return secure, existing media files for rendering on the track page."""
    assets: List[Dict[str, str]] = []
    seen_filenames = set()

    for raw_item in str(media_value or "").split(","):
        candidate = raw_item.strip()
        if not candidate:
            continue

        # Keep only plain filenames from our uploads folder.
        basename = os.path.basename(candidate)
        filename = secure_filename(basename)
        if not filename or filename != basename or filename in seen_filenames:
            continue

        extension = file_extension(filename)
        if extension not in ALLOWED_EXTENSIONS:
            continue

        absolute_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.isfile(absolute_path):
            continue

        media_type = "image" if extension in IMAGE_EXTENSIONS else "video"
        assets.append(
            {
                "filename": filename,
                "url": url_for("static", filename=f"uploads/{filename}"),
                "type": media_type,
                "mime": VIDEO_MIME_TYPES.get(extension, ""),
                "extension": extension.upper(),
            }
        )
        seen_filenames.add(filename)
        if len(assets) >= MAX_UPLOAD_FILES:
            break

    return assets


def enrich_track_complaint_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Attach proof metadata + progress fields for track page rendering."""
    media_value = str(row.get("media_paths") or row.get("photo") or "").strip()
    proof_files = build_proof_assets(media_value)
    row["proof_files"] = proof_files
    row["proof_count"] = len(proof_files)
    row["has_proof"] = len(proof_files) > 0
    progress_percent, progress_label = complaint_progress_details(row.get("status", ""))
    row["progress_percent"] = progress_percent
    row["progress_label"] = progress_label
    return row


def send_otp(email: str, otp: str) -> bool:
    """Send OTP via Brevo transactional email API."""
    try:
        if BREVO_API_KEY is not None and str(BREVO_API_KEY).strip():
            url = "https://api.brevo.com/v3/smtp/email"
            headers = {
                "accept": "application/json",
                "api-key": BREVO_API_KEY,
                "content-type": "application/json",
            }
            data = {
                "sender": {"email": BREVO_SENDER_EMAIL},
                "to": [{"email": email}],
                "subject": "OTP Verification",
                "htmlContent": f"<h3>Your OTP is {otp}</h3>",
            }

            response = requests.post(url, json=data, headers=headers, timeout=15)
            print("BREVO RESPONSE:", response.text)
            if response.status_code == 201:
                print("EMAIL SENT SUCCESS ✅")
                return True
            print("BREVO FAILED:", response.text)
            return False

        if not SMTP_HOST or not SMTP_USERNAME or not SMTP_PASSWORD:
            raise RuntimeError("SMTP settings are missing")

        sender = SMTP_FROM or SMTP_USERNAME
        if not sender:
            raise RuntimeError("SMTP sender is missing")

        try:
            smtp_port = int(str(SMTP_PORT).strip() or "587")
        except ValueError:
            smtp_port = 587

        use_tls = str(SMTP_USE_TLS).strip().lower() in {"1", "true", "yes", "on"}

        message = EmailMessage()
        message["Subject"] = "OTP Verification"
        message["From"] = sender
        message["To"] = email
        message.set_content(f"Your OTP is {otp}")
        message.add_alternative(f"<h3>Your OTP is {otp}</h3>", subtype="html")

        with smtplib.SMTP(SMTP_HOST, smtp_port, timeout=15) as smtp:
            if use_tls:
                smtp.starttls()
            smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            smtp.send_message(message)
        return True
    except Exception as e:
        print("OTP EMAIL ERROR:", e)
        return False


def build_resolution_email_html(
    payload: Dict[str, Any],
    tracking_url: str,
    feedback_url: str,
) -> str:
    """Return a branded HTML email for resolved complaints."""
    user_name = str(payload.get("user_name") or "Citizen")
    complaint_uid = str(payload.get("complaint_uid") or "")
    category = str(payload.get("category") or "")
    location = str(payload.get("location") or "")
    status = str(payload.get("status") or "Resolved")
    resolved_at = str(
        payload.get("resolved_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    department_name = str(
        payload.get("department_name")
        or CATEGORY_TO_DEPARTMENT.get(category, "Municipal Department")
    )

    raw_text = " ".join(str(payload.get("text") or "").split())
    summary = raw_text[:180] + ("..." if len(raw_text) > 180 else "")

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Complaint Resolved</title>
</head>
<body style=\"margin:0;padding:0;background:#f4f6fb;font-family:Arial,sans-serif;color:#0f172a;\">
    <table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" role=\"presentation\" style=\"background:#f4f6fb;padding:24px 12px;\">
        <tr>
            <td align=\"center\">
                <table width=\"600\" cellpadding=\"0\" cellspacing=\"0\" role=\"presentation\" style=\"width:100%;max-width:600px;background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 12px 30px rgba(15,23,42,0.12);\">
                    <tr>
                        <td style=\"background:linear-gradient(135deg,#0b3a6e,#1f6fb2);padding:20px 24px;color:#ffffff;\">
                            <div style=\"font-size:14px;letter-spacing:0.12em;text-transform:uppercase;opacity:0.8;\">Smart Grievance</div>
                            <div style=\"font-size:22px;font-weight:700;margin-top:6px;\">Your Complaint Has Been Resolved</div>
                        </td>
                    </tr>
                    <tr>
                        <td style=\"padding:24px;\">
                            <p style=\"margin:0 0 12px;font-size:16px;\">Hello {user_name},</p>
                            <p style=\"margin:0 0 16px;font-size:15px;line-height:1.6;\">
                                Your complaint has been successfully resolved. Here are the details for your records.
                            </p>

                            <div style=\"margin:0 0 18px;padding:14px 16px;border-radius:12px;background:#f8fafc;border:1px solid #e2e8f0;\">
                                <div style=\"display:flex;flex-wrap:wrap;gap:10px;align-items:center;\">
                                    <div style=\"font-size:13px;color:#64748b;\">Status</div>
                                    <span style=\"display:inline-block;padding:4px 10px;border-radius:999px;background:#dcfce7;color:#166534;font-weight:700;font-size:12px;\">{status}</span>
                                    <span style=\"margin-left:auto;font-size:12px;color:#64748b;\">Resolved at {resolved_at}</span>
                                </div>
                            </div>

                            <table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" role=\"presentation\" style=\"font-size:14px;line-height:1.6;\">
                                <tr>
                                    <td style=\"padding:6px 0;color:#64748b;width:160px;\">Complaint ID</td>
                                    <td style=\"padding:6px 0;font-weight:600;\">{complaint_uid}</td>
                                </tr>
                                <tr>
                                    <td style=\"padding:6px 0;color:#64748b;\">Department</td>
                                    <td style=\"padding:6px 0;font-weight:600;\">{department_name}</td>
                                </tr>
                                <tr>
                                    <td style=\"padding:6px 0;color:#64748b;\">Category</td>
                                    <td style=\"padding:6px 0;font-weight:600;\">{category}</td>
                                </tr>
                                <tr>
                                    <td style=\"padding:6px 0;color:#64748b;\">Location</td>
                                    <td style=\"padding:6px 0;font-weight:600;\">{location}</td>
                                </tr>
                                <tr>
                                    <td style=\"padding:6px 0;color:#64748b;\">Summary</td>
                                    <td style=\"padding:6px 0;\">{summary}</td>
                                </tr>
                            </table>

                            <div style=\"margin-top:22px;\">
                                <a href=\"{tracking_url}\" style=\"display:inline-block;background:#1f6fb2;color:#ffffff;text-decoration:none;padding:10px 18px;border-radius:10px;font-weight:600;font-size:14px;\">Track Complaint</a>
                                <a href=\"{feedback_url}\" style=\"display:inline-block;margin-left:10px;background:#ffffff;color:#1f6fb2;text-decoration:none;padding:10px 18px;border-radius:10px;font-weight:600;font-size:14px;border:1px solid #1f6fb2;\">Share Feedback</a>
                            </div>

                            <p style=\"margin:20px 0 0;font-size:13px;color:#64748b;\">
                                Thank you for using the Smart Grievance Management System.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""


def send_resolution_email(payload: Dict[str, Any], base_url: str) -> bool:
    """Send a complaint resolution email using Brevo."""
    try:
        receiver = str(payload.get("user_email") or "").strip()
        if not receiver:
            raise RuntimeError("User email is missing")

        user_name = str(payload.get("user_name") or "Citizen")
        complaint_uid = str(payload.get("complaint_uid") or "")
        resolved_at = str(
            payload.get("resolved_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        department_name = (
            payload.get("department_name")
            or CATEGORY_TO_DEPARTMENT.get(payload.get("category") or "", "Municipal Department")
        )
        tracking_url = f"{base_url.rstrip('/')}/track?q={complaint_uid}"
        feedback_url = f"{base_url.rstrip('/')}/feedback/{complaint_uid}"

        html_content = build_resolution_email_html(payload, tracking_url, feedback_url)
        text_content = "\n".join(
            [
                f"Hello {user_name},",
                "",
                "Your complaint has been successfully resolved.",
                "",
                f"Complaint ID: {complaint_uid}",
                f"Department: {department_name}",
                f"Location: {payload.get('location')}",
                f"Status: {payload.get('status')}",
                f"Resolved at: {resolved_at}",
                "",
                f"Track your complaint: {tracking_url}",
                f"Share feedback: {feedback_url}",
                "",
                "Thank you for using Smart Grievance Management System.",
            ]
        )

        url = "https://api.brevo.com/v3/smtp/email"
        if BREVO_API_KEY and str(BREVO_API_KEY).strip() and BREVO_SENDER_EMAIL and str(BREVO_SENDER_EMAIL).strip():
            headers = {
                "accept": "application/json",
                "api-key": BREVO_API_KEY,
                "content-type": "application/json",
            }
            data = {
                "sender": {"email": BREVO_SENDER_EMAIL, "name": "Smart Grievance"},
                "to": [{"email": receiver, "name": user_name}],
                "subject": "Your Complaint Has Been Resolved",
                "htmlContent": html_content,
                "textContent": text_content,
            }

            response = requests.post(url, json=data, headers=headers, timeout=15)
            print("BREVO RESOLUTION RESPONSE:", response.text)
            if response.status_code == 201:
                return True

        if not SMTP_HOST or not SMTP_USERNAME or not SMTP_PASSWORD:
            raise RuntimeError("Brevo failed and SMTP settings are missing")

        sender = SMTP_FROM or SMTP_USERNAME
        if not sender:
            raise RuntimeError("SMTP sender is missing")

        try:
            smtp_port = int(str(SMTP_PORT).strip() or "587")
        except ValueError:
            smtp_port = 587

        use_tls = str(SMTP_USE_TLS).strip().lower() in {"1", "true", "yes", "on"}

        message = EmailMessage()
        message["Subject"] = "Your Complaint Has Been Resolved"
        message["From"] = sender
        message["To"] = receiver
        message.set_content(text_content)
        message.add_alternative(html_content, subtype="html")

        with smtplib.SMTP(SMTP_HOST, smtp_port, timeout=15) as smtp:
            if use_tls:
                smtp.starttls()
            smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            smtp.send_message(message)
        return True
    except Exception as exc:
        print("RESOLUTION EMAIL ERROR:", exc)
        return False


# =============================================================================
# LAYOUT — shared variables for templates/base.html (navbar + sidebar)
# =============================================================================


def _sidebar_home_link_and_title(
    department_slug: Optional[str], admin: bool
) -> Tuple[str, str]:
    """(URL for sidebar home, short title for current panel)."""
    if department_slug is not None:
        return url_for("dashboard", department=department_slug), department_panel_title(
            department_slug
        )
    if admin:
        return url_for("admin_dashboard"), "Admin Control Room"
    return url_for("index"), "Citizen Portal"


def _navbar_user_labels_and_actions(
    department_slug: Optional[str], admin: bool
) -> Dict[str, str]:
    """Display name, role, logout (or login) target — keys match template variable names."""
    if admin:
        return {
            "shell_user_name": "System Administrator",
            "shell_user_role": "Control Room Admin",
            "shell_logout_url": url_for("admin_logout"),
            "shell_logout_label": "Logout",
        }
    if department_slug is not None:
        username = str(session.get("username", "Department Officer"))
        category = DEPARTMENT_TO_CATEGORY.get(department_slug, "Department")
        return {
            "shell_user_name": username.replace("_", " ").title(),
            "shell_user_role": f"{category} Officer",
            "shell_logout_url": url_for("logout"),
            "shell_logout_label": "Logout",
        }
    if is_user_logged_in():
        display = str(session.get("username", "Citizen User"))
        return {
            "shell_user_name": display,
            "shell_user_role": "Citizen Account",
            "shell_logout_url": url_for("user_logout"),
            "shell_logout_label": "Logout",
        }
    return {
        "shell_user_name": "Public User",
        "shell_user_role": "Citizen Access",
        "shell_logout_url": url_for("user_login"),
        "shell_logout_label": "User Login",
    }


def _sidebar_pending_badge_count(
    department_slug: Optional[str], admin: bool
) -> int:
    """Sidebar badge count for track menu (pending for staff/admin, total for citizens)."""
    try:
        if department_slug is not None:
            category = DEPARTMENT_TO_CATEGORY.get(department_slug)
            analytics = get_analytics_data(category=category) if category else {}
            return int(analytics.get("pending_total", 0))
        elif admin:
            analytics = get_analytics_data()
            return int(analytics.get("pending_total", 0))
        elif is_user_logged_in():
            return count_user_complaints(int(session["user_id"]))
        else:
            return 0
    except Exception:
        return 0


@app.context_processor
def inject_layout_context():
    """Flask calls this automatically before rendering any template."""
    department_slug = current_department_from_session()
    admin = is_admin_logged_in()
    citizen_user = is_user_logged_in() and department_slug is None and not admin

    dashboard_url, department_name = _sidebar_home_link_and_title(department_slug, admin)
    shell = _navbar_user_labels_and_actions(department_slug, admin)

    display_name = shell["shell_user_name"]
    initials = "".join(part[0].upper() for part in display_name.split()[:2] if part) or "SU"

    return {
        "active_department_slug": department_slug,
        "active_department_name": department_name,
        "is_admin_user": admin,
        "is_citizen_user": citizen_user,
        "sidebar_dashboard_url": dashboard_url,
        "sidebar_pending_count": _sidebar_pending_badge_count(department_slug, admin),
        "shell_user_name": shell["shell_user_name"],
        "shell_user_role": shell["shell_user_role"],
        "shell_user_initials": initials,
        "shell_logout_url": shell["shell_logout_url"],
        "shell_logout_label": shell["shell_logout_label"],
    }


@app.context_processor
def inject_lang_context():
    return {
        "get_text": get_text,
        "current_lang": session.get("lang", "en"),
    }


@app.errorhandler(RequestEntityTooLarge)
def handle_upload_too_large(error):
    del error
    flash(
        f"Uploads exceed the {MAX_REQUEST_SIZE_MB} MB limit. Please upload smaller files."
    )
    target = url_for("index") if is_user_logged_in() else url_for("user_login")
    return redirect(target)


# =============================================================================
# ROUTES — URL map (method → view function)
# =============================================================================
#   GET  /                  → home               user login/signup entry
#   GET  /complaint         → index              citizen complaint form
#   POST /submit            → submit_complaint   save complaint + ML category
#   GET  /track             → track_complaint    lookup form
#   POST /track             → track_complaint    lookup by UID
#   GET  /auth              → auth               department/admin login form
#   POST /auth/login        → auth_login         department/admin login
#   GET  /logout            → logout             clear department session
#   GET  /admin/logout      → admin_logout
#   GET  /admin             → admin_dashboard    all complaints (admin only)
#   GET  /dashboard/<slug>  → dashboard          one department (staff only)
#   POST /update_status     → update_status      change complaint status
#   GET  /analytics         → analytics_dashboard
# =============================================================================


@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("user_login"))


@app.route("/complaint", methods=["GET"])
def index():
    if not is_user_logged_in():
        return redirect(url_for("user_login"))
    return render_template("index.html", title="Register Complaint")


@app.route("/user/login", methods=["GET", "POST"])
def user_login():
    if request.method == "POST":
        identifier = request.form.get("login_id", "").strip()
        password = request.form.get("password", "").strip()

        account = get_user_by_login(identifier)
        if not account or not check_password_hash(account["password_hash"], password):
            flash("Invalid username/email or password.")
            return redirect(url_for("user_login"))

        if not int(account["is_verified"] or 0):
            session["pending_user_id"] = int(account["id"])
            flash("Please verify your email to continue.")
            return redirect(url_for("verify_email"))

        session.pop("department", None)
        session.pop("is_admin", None)
        session.pop("pending_user_id", None)
        session["user_id"] = int(account["id"])
        session["username"] = str(account["username"])
        flash("Login successful", "success")
        return redirect(url_for("index"))

    return render_template("user_login.html", title="User Login", show_register=False)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not username or not email or not password or not confirm_password:
            flash("All fields are required.")
            return redirect(url_for("register"))

        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            flash("Please enter a valid email address.")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for("register"))

        password_hash = generate_password_hash(password)
        otp_code = str(random.randint(100000, 999999))
        otp_expiry = (datetime.now() + timedelta(minutes=5)).isoformat()
        created, user_id, message = create_user_account(
            username=username,
            email=email,
            password_hash=password_hash,
            otp_code=otp_code,
            otp_expiry=otp_expiry,
        )
        if not created:
            conflict_identifier = username if message == "Username already exists." else email
            existing_user = get_user_by_login(conflict_identifier)

            if existing_user and not int(existing_user["is_verified"] or 0):
                session["pending_user_id"] = int(existing_user["id"])
                existing_email = str(existing_user["email"] or email)
                otp_code_existing = str(existing_user["otp_code"] or "").strip()
                otp_expiry_raw = str(existing_user["otp_expiry"] or "").strip()
                otp_still_valid = False
                if otp_code_existing and otp_expiry_raw:
                    try:
                        otp_expiry_existing = datetime.fromisoformat(otp_expiry_raw)
                        otp_still_valid = datetime.now() <= otp_expiry_existing
                    except ValueError:
                        otp_still_valid = False

                if otp_still_valid:
                    flash("OTP already sent. Please check your email or resend.", "success")
                else:
                    otp_code = str(random.randint(100000, 999999))
                    otp_expiry = (datetime.now() + timedelta(minutes=5)).isoformat()
                    update_user_otp(int(existing_user["id"]), otp_code, otp_expiry)
                    if send_otp(existing_email, otp_code):
                        flash("Account exists but is not verified. OTP resent.", "success")
                    else:
                        flash(
                            "Failed to send OTP. Please check email configuration.",
                            "danger",
                        )
                return redirect(url_for("verify_email"))

            session.pop("pending_user_id", None)
            flash("Account already exists, please login")
            return redirect(url_for("user_login"))

        if user_id is None:
            flash("Registration failed. Please try again.")
            return redirect(url_for("register"))

        session["pending_user_id"] = int(user_id)
        if send_otp(email, otp_code):
            flash("OTP sent to your email.", "success")
        else:
            flash("Failed to send OTP. Please check email configuration.", "danger")
        return redirect(url_for("verify_email"))

    return render_template("user_login.html", title="Register", show_register=True)


@app.route("/auth", methods=["GET"])
def auth():
    active_department = current_department_from_session()
    if active_department:
        return redirect(url_for("dashboard", department=active_department))
    if is_admin_logged_in():
        return redirect(url_for("admin_dashboard"))
    return render_template("auth.html", title="Staff Login")


@app.route("/login", methods=["GET"])
def legacy_login():
    return redirect(url_for("auth"))


@app.route("/admin/login", methods=["GET"])
def legacy_admin_login():
    return redirect(url_for("auth"))


@app.route("/auth/login", methods=["POST"])
def auth_login():
    login_type = request.form.get("login_type", "").strip()
    session.clear()

    if login_type == "department":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        record = DEPARTMENT_USERS.get(username)

        if not record or record["password"] != password:
            flash("Invalid department credentials.")
            return redirect(url_for("auth"))

        dept_slug = normalize_department_slug(record["department"])
        session["username"] = username
        session["department"] = dept_slug
        return redirect(url_for("dashboard", department=dept_slug))

    if login_type == "admin":
        password = request.form.get("admin_password", "").strip()
        if password != ADMIN_PASSWORD:
            flash("Invalid admin password.")
            return redirect(url_for("auth"))

        session["is_admin"] = True
        return redirect(url_for("admin_dashboard"))

    flash("Invalid login request.")
    return redirect(url_for("auth"))


@app.route("/admin/logout", methods=["GET"])
def admin_logout():
    session.pop("is_admin", None)
    flash("Admin logged out.")
    return redirect(url_for("auth"))


@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    flash("Logged out.")
    return redirect(url_for("auth"))


@app.route("/user/logout", methods=["GET"])
def user_logout():
    session.pop("user_id", None)
    session.pop("username", None)
    session.pop("pending_user_id", None)
    flash("Logged out.")
    return redirect(url_for("user_login"))


@app.route("/verify-email", methods=["GET", "POST"])
def verify_email():
    pending_id = session.get("pending_user_id")
    if not pending_id:
        flash("Please login or register to verify your email.")
        return redirect(url_for("user_login"))

    user = get_user_by_id(int(pending_id))
    if user is None:
        session.pop("pending_user_id", None)
        flash("User not found. Please register again.")
        return redirect(url_for("register"))

    if int(user["is_verified"] or 0):
        session.pop("pending_user_id", None)
        session.pop("department", None)
        session.pop("is_admin", None)
        session["user_id"] = int(user["id"])
        session["username"] = str(user["username"])
        flash("Email already verified. Logged in.", "success")
        return redirect(url_for("index"))

    if request.method == "POST":
        otp_code = request.form.get("otp_code", "").strip()
        expiry_raw = str(user["otp_expiry"] or "")
        if not otp_code or not expiry_raw:
            flash("Invalid OTP. Please try again.")
            return redirect(url_for("verify_email"))

        try:
            expiry = datetime.fromisoformat(expiry_raw)
        except ValueError:
            flash("OTP expired. Please request a new one.")
            return redirect(url_for("verify_email"))

        if datetime.now() > expiry:
            flash("OTP expired. Please request a new one.")
            return redirect(url_for("verify_email"))

        if otp_code != str(user["otp_code"] or ""):
            flash("Incorrect OTP. Please try again.")
            return redirect(url_for("verify_email"))

        mark_user_verified(int(user["id"]))
        session.pop("pending_user_id", None)
        session.pop("department", None)
        session.pop("is_admin", None)
        session["user_id"] = int(user["id"])
        session["username"] = str(user["username"])
        flash("Email verified successfully", "success")
        return redirect(url_for("index"))

    return render_template("verify_email.html", title="Verify Email", email=user["email"])


@app.route("/resend-otp", methods=["POST"])
def resend_otp():
    pending_id = session.get("pending_user_id")
    if not pending_id:
        flash("Please login or register to verify your email.")
        return redirect(url_for("user_login"))

    user = get_user_by_id(int(pending_id))
    if user is None:
        session.pop("pending_user_id", None)
        flash("User not found. Please register again.")
        return redirect(url_for("register"))

    if int(user["is_verified"] or 0):
        session.pop("pending_user_id", None)
        flash("Email already verified. Please login.")
        return redirect(url_for("user_login"))

    otp_code = str(random.randint(100000, 999999))
    otp_expiry = (datetime.now() + timedelta(minutes=5)).isoformat()
    update_user_otp(int(user["id"]), otp_code, otp_expiry)

    if send_otp(str(user["email"]), otp_code):
        flash("OTP resent to your email.", "success")
    else:
        flash("Failed to send OTP. Please check email configuration.", "danger")
    return redirect(url_for("verify_email"))


@app.route("/set-language/<lang>")
def set_language(lang: str):
    if lang in {"en", "hi"}:
        session["lang"] = lang
    return redirect(request.referrer or url_for("index"))


@app.route("/geocode/reverse", methods=["POST"])
def reverse_geocode():
    if not is_user_logged_in():
        return jsonify({"error": "Unauthorized"}), 403

    payload = request.get_json(silent=True) or {}
    lat = parse_coordinate(
        payload.get("latitude") or request.form.get("latitude"),
        -90.0,
        90.0,
    )
    lon = parse_coordinate(
        payload.get("longitude") or request.form.get("longitude"),
        -180.0,
        180.0,
    )

    if lat is None or lon is None:
        return jsonify({"error": "Invalid coordinates."}), 400

    params = {
        "format": "jsonv2",
        "lat": lat,
        "lon": lon,
        "zoom": 14,
        "addressdetails": 1,
    }
    headers = {"User-Agent": "SmartGrievance/1.0"}

    try:
        response = requests.get(
            REVERSE_GEOCODE_URL,
            params=params,
            headers=headers,
            timeout=REVERSE_GEOCODE_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return jsonify({"error": "Location lookup failed."}), 502
        data = response.json()
    except requests.RequestException:
        return jsonify({"error": "Location lookup failed."}), 502

    address = data.get("address") if isinstance(data, dict) else None
    if not isinstance(address, dict):
        address = {}

    address_text = " ".join(str(value) for value in address.values() if value)
    area = detect_area_from_message(address_text)
    city = (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("county")
        or address.get("state_district")
    )
    fallback = str(data.get("display_name") or address_text or "").strip()
    location_label = build_location_label(area, city, fallback)

    if not location_label:
        return jsonify({"error": "Location not found."}), 404

    return jsonify(
        {
            "location": location_label,
            "area": area or "",
            "city": city or "",
            "latitude": lat,
            "longitude": lon,
        }
    )


@app.route("/geocode/ip", methods=["GET"])
def ip_geocode():
    if not is_user_logged_in():
        return jsonify({"error": "Unauthorized"}), 403

    try:
        response = requests.get(IP_GEO_URL, timeout=IP_GEO_TIMEOUT_SECONDS)
        if response.status_code != 200:
            return jsonify({"error": "Approximate location failed."}), 502
        data = response.json()
    except requests.RequestException:
        return jsonify({"error": "Approximate location failed."}), 502

    if not isinstance(data, dict):
        return jsonify({"error": "Approximate location failed."}), 502

    lat = parse_coordinate(data.get("latitude"), -90.0, 90.0)
    lon = parse_coordinate(data.get("longitude"), -180.0, 180.0)
    if lat is None or lon is None:
        return jsonify({"error": "Approximate location not available."}), 404

    city = data.get("city")
    region = data.get("region") or data.get("region_code")
    country = data.get("country_name") or data.get("country")
    fallback = ", ".join(item for item in [city, region, country] if item)
    location_label = fallback[:LOCATION_LABEL_MAX_LEN] if fallback else ""

    if not location_label:
        return jsonify({"error": "Approximate location not available."}), 404

    return jsonify(
        {
            "location": location_label,
            "city": city or "",
            "latitude": lat,
            "longitude": lon,
            "source": "ip",
        }
    )


@app.route("/submit", methods=["POST"])
def submit_complaint():
    if not is_user_logged_in():
        return redirect(url_for("user_login"))
    name = str(session.get("username", "")).strip()
    location = request.form.get("location", "").strip()
    complaint_text = request.form.get("complaint_text", "").strip()
    latitude = parse_coordinate(request.form.get("latitude"), -90.0, 90.0)
    longitude = parse_coordinate(request.form.get("longitude"), -180.0, 180.0)

    if (latitude is None) != (longitude is None):
        latitude = None
        longitude = None

    if not complaint_text:
        flash("Please enter complaint.")
        return redirect(url_for("index"))
    if not name or not location:
        flash("Please fill out Name and Location.")
        return redirect(url_for("index"))

    user = get_user_by_id(int(session["user_id"]))
    if user is None:
        session.pop("user_id", None)
        session.pop("username", None)
        flash("Please log in again.")
        return redirect(url_for("user_login"))

    if not int(user["is_verified"] or 0):
        session["pending_user_id"] = int(user["id"])
        flash("Please verify your email to submit a complaint.")
        return redirect(url_for("verify_email"))

    uploads = request.files.getlist("media") or request.files.getlist("photos")
    files = [item for item in uploads if item and item.filename]
    if len(files) > MAX_UPLOAD_FILES:
        flash("Maximum 2 files allowed")
        return redirect(url_for("index"))

    for item in files:
        if not allowed_file(item.filename):
            flash("Only JPG, JPEG, PNG, MP4, MOV, and AVI files are allowed.")
            return redirect(url_for("index"))

    saved_files = []
    for item in files:
        extension = file_extension(item.filename)
        filename = secure_filename(f"{uuid4().hex}.{extension}")
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        item.save(file_path)

        if extension in VIDEO_EXTENSIONS:
            size_bytes = os.path.getsize(file_path)
            if size_bytes > MAX_VIDEO_SIZE_BYTES:
                cleanup_uploaded_files(saved_files + [filename])
                flash(f"Video must be under {MAX_VIDEO_SIZE_MB} MB.")
                return redirect(url_for("index"))
            if VideoFileClip is None:
                # Skip duration validation if MoviePy/ffmpeg is unavailable.
                pass
            else:
                clip = None
                try:
                    clip = VideoFileClip(file_path)
                    duration = float(clip.duration or 0)
                except Exception:
                    cleanup_uploaded_files(saved_files + [filename])
                    flash("Invalid video file uploaded.")
                    return redirect(url_for("index"))
                finally:
                    if clip is not None:
                        clip.close()

                if duration > MAX_VIDEO_DURATION_SECONDS:
                    cleanup_uploaded_files(saved_files + [filename])
                    flash("Video must be less than 1 minute")
                    return redirect(url_for("index"))

        saved_files.append(filename)

    media_paths = ",".join(saved_files) if saved_files else None

    user_id = int(user["id"])
    category = classify_complaint(complaint_text)
    priority = priority_from_text(complaint_text)
    location_key = normalize_location(location)
    complaint_uid, department_name, complaint_id = register_complaint(
        user_id=user_id,
        location=location,
        latitude=latitude,
        longitude=longitude,
        text=complaint_text,
        category=category,
        priority=priority,
        photo=media_paths,
        media_paths=media_paths,
        location_key=location_key,
    )

    auto_escalate_priority(
        complaint_id=complaint_id,
        location_key=location_key,
        category=category,
        complaint_text=complaint_text,
    )

    flash("Complaint registered successfully", "success")
    message = "Complaint registered successfully"
    return render_template(
        "result.html",
        title="Complaint Submitted",
        message=message,
        complaint_uid=complaint_uid,
        category=category,
        priority=priority,
        department=department_name,
        location=location,
    )


@app.route("/admin", methods=["GET"])
def admin_dashboard():
    """Full city view: every complaint; department users must not see this page."""
    if current_department_from_session() is not None:
        return "Unauthorized Access", 403
    if not is_admin_logged_in():
        return redirect(url_for("auth"))

    selected_category = request.args.get("category", "").strip()
    complaints = fetch_complaints(category=selected_category or None)
    categories = list(CATEGORY_TO_DEPARTMENT.keys())
    summary = dashboard_summary_counts(complaints)
    escalation_alerts = build_escalation_alerts([dict(row) for row in complaints])
    return render_template(
        "admin.html",
        title="Admin Dashboard",
        complaints=complaints,
        categories=categories,
        summary=summary,
        status_options=STATUS_FLOW,
        selected_category=selected_category,
        is_department_dashboard=False,
        dashboard_department="",
        dashboard_label="Smart Operations Dashboard",
        panel_label="City Grievance Command Panel",
        escalation_alerts=escalation_alerts,
    )


@app.route("/dashboard/<department>", methods=["GET"])
def dashboard(department: str):
    """Single-department view; URL slug must match the department saved at login."""
    assigned = current_department_from_session()
    if assigned is None:
        return redirect(url_for("auth"))

    requested = normalize_department_slug(department)
    if assigned != requested:
        return "Access Denied: Unauthorized Access", 403

    selected_category = DEPARTMENT_TO_CATEGORY.get(requested)
    if selected_category is None:
        return "Access Denied: Unauthorized Access", 403

    complaints = fetch_complaints(category=selected_category)
    categories = [selected_category]
    summary = dashboard_summary_counts(complaints)
    escalation_alerts = build_escalation_alerts([dict(row) for row in complaints])
    dashboard_label = f"{selected_category} Dashboard"
    return render_template(
        "admin.html",
        title=dashboard_label,
        complaints=complaints,
        categories=categories,
        summary=summary,
        status_options=STATUS_FLOW,
        selected_category=selected_category,
        is_department_dashboard=True,
        dashboard_department=requested,
        dashboard_label=dashboard_label,
        panel_label=department_panel_title(requested),
        escalation_alerts=escalation_alerts,
    )


@app.route("/admin/update-status/<complaint_uid>", methods=["POST"])
def update_status_legacy(complaint_uid: str):
    """Old URL kept on purpose; department accounts may not use it."""
    del complaint_uid
    if current_department_from_session() is None:
        return redirect(url_for("auth"))
    return "Unauthorized Access", 403


@app.route("/update_status", methods=["POST"])
def update_status():
    """Form post from dashboard: validate, then update row in database."""
    assigned = current_department_from_session()
    if assigned is None and not is_admin_logged_in():
        return redirect(url_for("auth"))

    complaint_id_raw = request.form.get("complaint_id", "").strip()
    new_status = request.form.get("status", "").strip()

    if assigned is None:
        redirect_target = request.referrer or url_for("admin_dashboard")
    else:
        redirect_target = url_for("dashboard", department=assigned)

    if new_status not in STATUS_FLOW:
        flash("Invalid status value.")
        return redirect(redirect_target)

    try:
        complaint_id = int(complaint_id_raw)
    except ValueError:
        flash("Invalid complaint ID.")
        return redirect(redirect_target)

    complaint = get_complaint_notification_payload(complaint_id)
    if complaint is None:
        flash("Complaint not found.")
        return redirect(redirect_target)

    # Department users may only touch complaints in their own category.
    if assigned is not None:
        selected_category = DEPARTMENT_TO_CATEGORY.get(assigned)
        if selected_category is None:
            return "Access Denied: Unauthorized Access", 403
        if complaint["category"] != selected_category:
            return "Unauthorized Access", 403

    previous_status = str(complaint["status"] or "")
    resolution_email_sent_at = str(complaint["resolution_email_sent_at"] or "").strip()

    updated = set_complaint_status_by_id(complaint_id=complaint_id, new_status=new_status)
    if updated is None:
        flash("Complaint not found.")
    else:
        flash(f"Status updated to {new_status}", "success")

        status_changed = previous_status != new_status
        should_notify = new_status == "Resolved" and status_changed and not resolution_email_sent_at
        if should_notify:
            refreshed = get_complaint_notification_payload(complaint_id)
            payload = dict(refreshed) if refreshed else None
            if payload and payload.get("status") == "Resolved":
                sent = send_resolution_email(payload, request.url_root)
                if sent:
                    sent_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    set_resolution_email_sent_at(complaint_id, sent_at)
                    flash("Resolution email sent successfully.", "success")
                else:
                    flash(
                        "Status updated, but resolution email failed to send.",
                        "danger",
                    )
    return redirect(redirect_target)


@app.route("/track", methods=["GET", "POST"])
def track_complaint():
    if not is_user_logged_in():
        return redirect(url_for("user_login"))

    user_id = int(session["user_id"])
    complaint = None
    searched_id = ""

    track_query = request.args.get("q", "").strip().upper()
    track_status = request.args.get("status", "").strip()
    track_category = request.args.get("category", "").strip()
    track_department = request.args.get("department", "").strip()

    complaints_rows = fetch_user_complaints(
        user_id=user_id,
        complaint_uid_query=track_query,
        status=track_status,
        category=track_category,
        department=track_department,
    )
    complaints = []
    for row in complaints_rows:
        complaints.append(enrich_track_complaint_record(dict(row)))

    complaints_total = count_user_complaints(user_id)

    if request.method == "POST":
        searched_id = request.form.get("complaint_uid", "").strip().upper()
        if searched_id:
            found = get_complaint_by_uid(searched_id)
            if found is not None and int(found["user_id"]) == user_id:
                complaint = enrich_track_complaint_record(dict(found))
                flash(f"Complaint {searched_id} found", "success")
            else:
                complaint = None
        if complaint is None:
            flash("Complaint ID not found")

    return render_template(
        "track.html",
        title="Track Complaint",
        complaint=complaint,
        complaints=complaints,
        complaints_total=complaints_total,
        track_status_options=TRACK_STATUS_OPTIONS,
        track_category_options=sorted(CATEGORY_TO_DEPARTMENT.keys()),
        track_department_options=sorted(set(CATEGORY_TO_DEPARTMENT.values())),
        track_filters={
            "q": track_query,
            "status": track_status,
            "category": track_category,
            "department": track_department,
        },
        searched_id=searched_id,
    )


@app.route("/feedback/<complaint_uid>", methods=["GET", "POST"])
def complaint_feedback(complaint_uid: str):
    normalized_uid = str(complaint_uid or "").strip().upper()
    complaint = get_complaint_by_uid(normalized_uid) if normalized_uid else None
    feedback = get_feedback_by_complaint_uid(normalized_uid) if normalized_uid else None
    feedback_saved = False
    feedback_error = ""

    if request.method == "POST":
        if feedback is not None:
            feedback_error = "Feedback already submitted."
        elif complaint is None:
            feedback_error = "Complaint not found."
        else:
            rating_raw = str(request.form.get("rating", "")).strip()
            message = str(request.form.get("message", "")).strip()
            try:
                rating = int(rating_raw)
            except ValueError:
                rating = None

            if rating is None or rating < 1 or rating > 5:
                feedback_error = "Please select a rating from 1 to 5."
            else:
                message = message[:500]
                saved, note = record_complaint_feedback(normalized_uid, rating, message)
                if saved:
                    feedback_saved = True
                    feedback = get_feedback_by_complaint_uid(normalized_uid)
                else:
                    feedback_error = note

    return render_template(
        "feedback.html",
        title="Complaint Feedback",
        complaint=complaint,
        feedback=feedback,
        feedback_saved=feedback_saved,
        feedback_error=feedback_error,
    )


@app.route("/chatbot/message", methods=["POST"])
def chatbot_message():
    if not is_user_logged_in() or is_admin_logged_in() or current_department_from_session() is not None:
        return jsonify({"error": "Unauthorized"}), 403

    if is_chatbot_rate_limited():
        return jsonify({"error": "Too many requests. Please wait a moment."}), 429

    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message") or request.form.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message is required."}), 400

    message = re.sub(r"[\x00-\x1f]", " ", message)
    message = message[:CHATBOT_MAX_MESSAGE_LEN]

    department = detect_department_from_message(message)
    area = detect_area_from_message(message)
    area_source = "message"

    if area is None:
        last_location = get_latest_user_location(int(session["user_id"]))
        area = resolve_area_from_location(last_location)
        area_source = "history" if area else ""

    if department is None:
        return jsonify(
            {
                "reply": {
                    "type": "text",
                    "text": (
                        "Tell me the department or issue type (water, electricity, road, garbage, "
                        "healthcare, fire)."
                    ),
                }
            }
        )

    if area is None:
        area_list = ", ".join(CHATBOT_AREAS)
        return jsonify(
            {
                "reply": {
                    "type": "text",
                    "text": f"Which area is this for? Available areas: {area_list}.",
                }
            }
        )

    contact = fetch_department_contact(department=department, area=area)
    if contact is None:
        return jsonify(
            {
                "reply": {
                    "type": "text",
                    "text": (
                        f"I could not find contact details for {department} in {area}. "
                        "Please confirm the area name."
                    ),
                }
            }
        )

    return jsonify(
        {
            "reply": {
                "type": "contact",
                "department": contact["department"],
                "area": contact["area"],
                "phone": contact["phone"],
                "email": contact["email"],
                "office": contact["office_address"],
                "area_source": area_source,
            }
        }
    )


@app.route("/analytics", methods=["GET"])
def analytics_dashboard():
    active_department = current_department_from_session()
    if active_department is None and not is_admin_logged_in():
        return redirect(url_for("auth"))

    analytics_scope = "All Departments"
    if active_department is not None:
        selected_category = DEPARTMENT_TO_CATEGORY.get(active_department)
        if selected_category is None:
            return "Access Denied: Unauthorized Access", 403
        analytics = get_analytics_data(category=selected_category)
        analytics_scope = f"{selected_category} Department"
    else:
        analytics = get_analytics_data()

    return render_template(
        "analytics.html",
        title="Analytics Dashboard",
        analytics=analytics,
        analytics_scope=analytics_scope,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=("cert.pem", "key.pem"))
