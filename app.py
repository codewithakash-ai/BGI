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

import os
import random
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from uuid import uuid4

import joblib
import requests
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from database.db import (
    CATEGORY_TO_DEPARTMENT,
    STATUS_FLOW,
    create_user_account,
    fetch_complaints,
    get_analytics_data,
    get_complaint_by_id,
    get_complaint_by_uid,
    get_user_by_id,
    get_user_by_login,
    init_db,
    mark_user_verified,
    register_complaint,
    set_complaint_status_by_id,
    update_user_otp,
)

# =============================================================================
# SETTINGS (demo values — replace for a real deployment)
# =============================================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = "citizen-grievance-demo-secret"
ADMIN_PASSWORD = "admin"

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024

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
    cleaned_text = re.sub(r"[^a-z\s]", " ", complaint_text.lower())
    tokens = set(cleaned_text.split())

    if tokens.intersection(HIGH_PRIORITY_KEYWORDS):
        return "High"
    if tokens.intersection(MEDIUM_PRIORITY_KEYWORDS):
        return "Medium"
    return "Low"


def classify_complaint(text: str) -> str:
    """Classify grievance text using pre-trained model + vectorizer."""
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    prediction_label = str(prediction).strip().lower()
    return MODEL_LABEL_TO_CATEGORY.get(prediction_label, "Roads")


def allowed_file(filename: str) -> bool:
    """Allow only specific image file extensions."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def dashboard_summary_counts(complaints) -> Dict[str, int]:
    """Numbers for the colored summary cards on admin / department dashboards."""
    return {
        "total": len(complaints),
        "pending": sum(1 for row in complaints if row["status"] == "Pending"),
        "in_progress": sum(1 for row in complaints if row["status"] == "In Progress"),
        "resolved": sum(1 for row in complaints if row["status"] == "Resolved"),
        "high_priority": sum(1 for row in complaints if row["priority"] == "High"),
    }


def send_otp(email: str, otp: str) -> bool:
    """Send OTP via Brevo transactional email API."""
    try:
        if BREVO_API_KEY is None or not str(BREVO_API_KEY).strip():
            raise RuntimeError("BREVO_API_KEY is missing")

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
        return response.status_code == 201
    except Exception as e:
        print("BREVO ERROR:", e)
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
    """Pending complaints count for sidebar badge; 0 if lookup fails or user is public."""
    try:
        if department_slug is not None:
            category = DEPARTMENT_TO_CATEGORY.get(department_slug)
            analytics = get_analytics_data(category=category) if category else {}
        elif admin:
            analytics = get_analytics_data()
        else:
            analytics = {}
        return int(analytics.get("pending_total", 0))
    except Exception:
        return 0


@app.context_processor
def inject_layout_context():
    """Flask calls this automatically before rendering any template."""
    department_slug = current_department_from_session()
    admin = is_admin_logged_in()

    dashboard_url, department_name = _sidebar_home_link_and_title(department_slug, admin)
    shell = _navbar_user_labels_and_actions(department_slug, admin)

    display_name = shell["shell_user_name"]
    initials = "".join(part[0].upper() for part in display_name.split()[:2] if part) or "SU"

    return {
        "active_department_slug": department_slug,
        "active_department_name": department_name,
        "is_admin_user": admin,
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


@app.route("/submit", methods=["POST"])
def submit_complaint():
    if not is_user_logged_in():
        return redirect(url_for("user_login"))
    name = str(session.get("username", "")).strip()
    location = request.form.get("location", "").strip()
    complaint_text = request.form.get("complaint_text", "").strip()

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

    uploads = request.files.getlist("photos")
    files = [item for item in uploads if item and item.filename]
    if len(files) > 2:
        flash("You can upload maximum 2 images only")
        return redirect(url_for("index"))

    for item in files:
        if not allowed_file(item.filename):
            flash("Only JPG and PNG images are allowed.")
            return redirect(url_for("index"))

    saved_files = []
    for item in files:
        extension = item.filename.rsplit(".", 1)[1].lower()
        filename = secure_filename(f"{uuid4().hex}.{extension}")
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        item.save(file_path)
        saved_files.append(filename)

    filename = ",".join(saved_files) if saved_files else None

    user_id = int(user["id"])
    category = classify_complaint(complaint_text)
    priority = priority_from_text(complaint_text)
    complaint_uid, department_name = register_complaint(
        user_id=user_id,
        location=location,
        text=complaint_text,
        category=category,
        priority=priority,
        photo=filename,
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

    complaint = get_complaint_by_id(complaint_id)
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

    updated = set_complaint_status_by_id(complaint_id=complaint_id, new_status=new_status)
    if updated is None:
        flash("Complaint not found.")
    else:
        flash(f"Status updated to {new_status}", "success")
    return redirect(redirect_target)


@app.route("/track", methods=["GET", "POST"])
def track_complaint():
    if not is_user_logged_in():
        return redirect(url_for("user_login"))
    complaint = None
    searched_id = ""
    if request.method == "POST":
        searched_id = request.form.get("complaint_uid", "").strip().upper()
        if searched_id:
            complaint = get_complaint_by_uid(searched_id)
        if complaint is not None:
            flash(f"Complaint {searched_id} found", "success")
        if complaint is None:
            flash("Complaint ID not found")
    return render_template(
        "track.html",
        title="Track Complaint",
        complaint=complaint,
        searched_id=searched_id,
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
