"""SQLite data access layer for the grievance management system."""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "grievance_system.db")

CATEGORY_TO_DEPARTMENT = {
    "Water": "Water Department",
    "Electricity": "Electricity Department",
    "Roads": "Municipal Department",
    "Garbage": "Sanitation Department",
    "Healthcare": "Health Department",
    "Fire": "Fire Department",
}

CONTACT_AREAS = [
    "Misrod",
    "MP Nagar",
    "Danish Nagar",
    "Indus",
    "Ayodhya Nagar",
    "Kolar",
    "BHEL",
    "Arera Colony",
]

CONTACT_DEPARTMENTS = [
    {
        "name": "Water Department",
        "phone_prefix": "1800-111",
        "email_prefix": "water",
        "office_suffix": "Water Office",
    },
    {
        "name": "Electricity Department",
        "phone_prefix": "1800-222",
        "email_prefix": "electricity",
        "office_suffix": "Electricity Office",
    },
    {
        "name": "Road/Municipal Department",
        "phone_prefix": "1800-333",
        "email_prefix": "roads",
        "office_suffix": "Municipal Office",
    },
    {
        "name": "Garbage Department",
        "phone_prefix": "1800-444",
        "email_prefix": "garbage",
        "office_suffix": "Garbage Office",
    },
    {
        "name": "Healthcare Department",
        "phone_prefix": "1800-555",
        "email_prefix": "healthcare",
        "office_suffix": "Healthcare Office",
    },
    {
        "name": "Fire Department",
        "phone_prefix": "1800-666",
        "email_prefix": "fire",
        "office_suffix": "Fire Office",
    },
]

STATUS_FLOW = ["Pending", "In Progress", "Resolved"]
PRIORITY_FLOW = ["High", "Medium", "Low"]


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with dict-like row access."""
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    """Create tables and seed departments if they do not exist."""
    max_attempts = 3
    retry_delay_seconds = 0.5

    for attempt in range(1, max_attempts + 1):
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    username TEXT,
                    email TEXT,
                    password_hash TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            _ensure_user_auth_columns(conn)
            _ensure_user_unique_indexes(conn)

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS complaints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    complaint_uid TEXT UNIQUE NOT NULL,
                    user_id INTEGER NOT NULL,
                    location TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    text TEXT NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'Pending',
                    priority TEXT NOT NULL DEFAULT 'Low',
                    photo TEXT,
                    media_paths TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
                """
            )
            _ensure_status_column(conn)
            _ensure_priority_column(conn)
            _ensure_resolved_at_column(conn)
            _ensure_resolution_email_sent_at_column(conn)
            _ensure_photo_column(conn)
            _ensure_media_paths_column(conn)
            _ensure_location_key_column(conn)
            _ensure_location_coords_columns(conn)
            _ensure_escalation_columns(conn)
            _ensure_department_contacts_table(conn)
            _ensure_feedback_table(conn)

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS departments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    complaint_id INTEGER NOT NULL UNIQUE,
                    department_id INTEGER NOT NULL,
                    FOREIGN KEY (complaint_id) REFERENCES complaints (id),
                    FOREIGN KEY (department_id) REFERENCES departments (id)
                )
                """
            )

            _seed_departments(conn)
            _seed_department_contacts(conn)
            conn.commit()
            return
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if conn is not None:
                conn.rollback()
            is_locked = "database is locked" in message or "database table is locked" in message
            if is_locked and attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            raise
        finally:
            if conn is not None:
                conn.close()


def _ensure_user_auth_columns(conn: sqlite3.Connection) -> None:
    """Safely add authentication columns for the users table if missing."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}

    if "username" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN username TEXT")
    if "password_hash" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    if "created_at" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
    if "is_verified" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN is_verified INTEGER")
    if "otp_code" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN otp_code TEXT")
    if "otp_expiry" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN otp_expiry TEXT")
    conn.execute(
        "UPDATE users SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
    )
    conn.execute(
        """
        UPDATE users
        SET is_verified = CASE
            WHEN password_hash IS NOT NULL AND otp_code IS NULL THEN 1
            ELSE 0
        END
        WHERE is_verified IS NULL
        """
    )


def _ensure_user_unique_indexes(conn: sqlite3.Connection) -> None:
    """Add unique indexes for username/email for auth accounts."""
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)"
    )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")


def _ensure_status_column(conn: sqlite3.Connection) -> None:
    """Safely add status column for older complaint tables if missing."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(complaints)").fetchall()
    }
    if "status" not in columns:
        conn.execute(
            "ALTER TABLE complaints ADD COLUMN status TEXT NOT NULL DEFAULT 'Pending'"
        )
    conn.execute(
        "UPDATE complaints SET status = 'Pending' WHERE status IS NULL OR TRIM(status) = ''"
    )


def _ensure_priority_column(conn: sqlite3.Connection) -> None:
    """Safely add priority column for older complaint tables if missing."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(complaints)").fetchall()
    }
    if "priority" not in columns:
        conn.execute(
            "ALTER TABLE complaints ADD COLUMN priority TEXT NOT NULL DEFAULT 'Low'"
        )
    conn.execute(
        "UPDATE complaints SET priority = 'Low' WHERE priority IS NULL OR TRIM(priority) = ''"
    )


def _ensure_resolved_at_column(conn: sqlite3.Connection) -> None:
    """Safely add resolved timestamp column and backfill resolved rows."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(complaints)").fetchall()
    }
    if "resolved_at" not in columns:
        conn.execute("ALTER TABLE complaints ADD COLUMN resolved_at TEXT")

    conn.execute(
        """
        UPDATE complaints
        SET resolved_at = created_at
        WHERE status = 'Resolved' AND (resolved_at IS NULL OR TRIM(resolved_at) = '')
        """
    )
    conn.execute(
        """
        UPDATE complaints
        SET resolved_at = NULL
        WHERE status != 'Resolved'
        """
    )


def _ensure_resolution_email_sent_at_column(conn: sqlite3.Connection) -> None:
    """Safely add resolution email timestamp column if missing."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(complaints)").fetchall()
    }
    if "resolution_email_sent_at" not in columns:
        conn.execute(
            "ALTER TABLE complaints ADD COLUMN resolution_email_sent_at TEXT"
        )


def _ensure_photo_column(conn: sqlite3.Connection) -> None:
    """Safely add photo column for complaint evidence if missing."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(complaints)").fetchall()
    }
    if "photo" not in columns:
        conn.execute("ALTER TABLE complaints ADD COLUMN photo TEXT")


def _ensure_media_paths_column(conn: sqlite3.Connection) -> None:
    """Safely add media_paths column and backfill legacy photo values."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(complaints)").fetchall()
    }
    if "media_paths" not in columns:
        conn.execute("ALTER TABLE complaints ADD COLUMN media_paths TEXT")

    conn.execute(
        """
        UPDATE complaints
        SET media_paths = photo
        WHERE (media_paths IS NULL OR TRIM(media_paths) = '')
          AND photo IS NOT NULL
          AND TRIM(photo) != ''
        """
    )
    conn.execute(
        """
        UPDATE complaints
        SET photo = media_paths
        WHERE (photo IS NULL OR TRIM(photo) = '')
          AND media_paths IS NOT NULL
          AND TRIM(media_paths) != ''
        """
    )


def _ensure_location_key_column(conn: sqlite3.Connection) -> None:
    """Safely add location_key column and create a lookup index."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(complaints)").fetchall()
    }
    if "location_key" not in columns:
        conn.execute("ALTER TABLE complaints ADD COLUMN location_key TEXT")
    conn.execute(
        """
        UPDATE complaints
        SET location_key = LOWER(TRIM(REPLACE(REPLACE(location, ',', ' '), '.', ' ')))
        WHERE location_key IS NULL OR TRIM(location_key) = ''
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_complaints_location_key ON complaints(location_key)"
    )


def _ensure_location_coords_columns(conn: sqlite3.Connection) -> None:
    """Safely add latitude/longitude columns for complaints if missing."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(complaints)").fetchall()
    }
    if "latitude" not in columns:
        conn.execute("ALTER TABLE complaints ADD COLUMN latitude REAL")
    if "longitude" not in columns:
        conn.execute("ALTER TABLE complaints ADD COLUMN longitude REAL")


def _ensure_escalation_columns(conn: sqlite3.Connection) -> None:
    """Safely add escalation metadata columns."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(complaints)").fetchall()
    }
    if "is_escalated" not in columns:
        conn.execute(
            "ALTER TABLE complaints ADD COLUMN is_escalated INTEGER DEFAULT 0"
        )
    if "escalation_reason" not in columns:
        conn.execute("ALTER TABLE complaints ADD COLUMN escalation_reason TEXT")
    if "escalation_group" not in columns:
        conn.execute("ALTER TABLE complaints ADD COLUMN escalation_group TEXT")
    if "escalation_count" not in columns:
        conn.execute("ALTER TABLE complaints ADD COLUMN escalation_count INTEGER")
    if "escalated_at" not in columns:
        conn.execute("ALTER TABLE complaints ADD COLUMN escalated_at TEXT")


def _ensure_department_contacts_table(conn: sqlite3.Connection) -> None:
    """Ensure the department_contacts table exists."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS department_contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department TEXT NOT NULL,
            area TEXT NOT NULL,
            phone TEXT NOT NULL,
            email TEXT NOT NULL,
            office_address TEXT NOT NULL,
            UNIQUE(department, area)
        )
        """
    )


def _ensure_feedback_table(conn: sqlite3.Connection) -> None:
    """Ensure complaint feedback table exists."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS complaint_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id INTEGER NOT NULL UNIQUE,
            complaint_uid TEXT NOT NULL,
            user_id INTEGER,
            rating INTEGER,
            message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (complaint_id) REFERENCES complaints (id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )


def _seed_department_contacts(conn: sqlite3.Connection) -> None:
    """Insert baseline contact data for each department and area."""
    cursor = conn.cursor()
    for dept in CONTACT_DEPARTMENTS:
        for index, area in enumerate(CONTACT_AREAS, start=1):
            area_key = area.lower().replace(" ", "").replace("-", "")
            phone = f"{dept['phone_prefix']}-{200 + index:03d}"
            email = f"{dept['email_prefix']}.{area_key}@gov.in"
            office = f"{area} {dept['office_suffix']}"
            cursor.execute(
                """
                INSERT OR IGNORE INTO department_contacts
                    (department, area, phone, email, office_address)
                VALUES (?, ?, ?, ?, ?)
                """,
                (dept["name"], area, phone, email, office),
            )


def _seed_departments(conn: sqlite3.Connection) -> None:
    """Insert predefined departments used by routing logic."""
    cursor = conn.cursor()
    for department_name in set(CATEGORY_TO_DEPARTMENT.values()):
        cursor.execute(
            "INSERT OR IGNORE INTO departments (name) VALUES (?)",
            (department_name,),
        )


def get_or_create_user(name: str, email: Optional[str] = None) -> int:
    """Find a user by (name, email) or create a new one."""
    conn = get_connection()
    cursor = conn.cursor()

    if email:
        row = cursor.execute(
            "SELECT id FROM users WHERE name = ? AND email = ?",
            (name, email),
        ).fetchone()
    else:
        row = cursor.execute(
            "SELECT id FROM users WHERE name = ? AND email IS NULL",
            (name,),
        ).fetchone()

    if row:
        conn.close()
        return int(row["id"])

    cursor.execute(
        "INSERT INTO users (name, email) VALUES (?, ?)",
        (name, email),
    )
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return int(user_id)


def create_user_account(
    username: str,
    email: str,
    password_hash: str,
    otp_code: str,
    otp_expiry: str,
    is_verified: int = 0,
) -> Tuple[bool, Optional[int], str]:
    """Create an auth user account if username/email are unique."""
    conn = get_connection()
    cursor = conn.cursor()

    existing = cursor.execute(
        "SELECT id, username, email FROM users WHERE username = ? OR email = ?",
        (username, email),
    ).fetchone()
    if existing:
        conn.close()
        if existing["username"] == username:
            return False, None, "Username already exists."
        return False, None, "Email already exists."

    cursor.execute(
        """
        INSERT INTO users (name, username, email, password_hash, is_verified, otp_code, otp_expiry)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (username, username, email, password_hash, is_verified, otp_code, otp_expiry),
    )
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return True, int(user_id), "Registration successful."


def get_user_by_login(identifier: str) -> Optional[sqlite3.Row]:
    """Fetch a user by username or email for login."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ? OR email = ?",
        (identifier, identifier),
    ).fetchone()
    conn.close()
    return row


def get_user_by_id(user_id: int) -> Optional[sqlite3.Row]:
    """Fetch a user by id."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    return row


def get_latest_user_location(user_id: int) -> Optional[str]:
    """Return the most recent complaint location for a user."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT location
        FROM complaints
        WHERE user_id = ? AND location IS NOT NULL AND TRIM(location) != ''
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return str(row["location"] or "").strip() or None


def fetch_department_contact(department: str, area: str) -> Optional[sqlite3.Row]:
    """Fetch a single department contact row by department + area."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT department, area, phone, email, office_address
        FROM department_contacts
        WHERE department = ? AND area = ?
        """,
        (department, area),
    ).fetchone()
    conn.close()
    return row


def update_user_otp(user_id: int, otp_code: str, otp_expiry: str) -> None:
    """Update OTP code and expiry for a user."""
    conn = get_connection()
    conn.execute(
        "UPDATE users SET otp_code = ?, otp_expiry = ? WHERE id = ?",
        (otp_code, otp_expiry, user_id),
    )
    conn.commit()
    conn.close()


def mark_user_verified(user_id: int) -> None:
    """Mark user as verified and clear OTP fields."""
    conn = get_connection()
    conn.execute(
        "UPDATE users SET is_verified = 1, otp_code = NULL, otp_expiry = NULL WHERE id = ?",
        (user_id,),
    )
    conn.commit()
    conn.close()


def _generate_unique_complaint_uid(conn: sqlite3.Connection) -> str:
    """Generate a unique complaint tracking ID."""
    while True:
        date_prefix = datetime.now().strftime("%Y%m%d")
        complaint_uid = f"CMP-{date_prefix}-{uuid.uuid4().hex[:6].upper()}"
        existing = conn.execute(
            "SELECT id FROM complaints WHERE complaint_uid = ?",
            (complaint_uid,),
        ).fetchone()
        if existing is None:
            return complaint_uid


def register_complaint(
    user_id: int,
    location: str,
    latitude: Optional[float],
    longitude: Optional[float],
    text: str,
    category: str,
    status: str = "Pending",
    priority: str = "Low",
    photo: Optional[str] = None,
    media_paths: Optional[str] = None,
    location_key: Optional[str] = None,
) -> Tuple[str, str, int]:
    """Insert complaint and assign it to mapped department."""
    conn = get_connection()
    cursor = conn.cursor()
    sanitized_priority = priority if priority in PRIORITY_FLOW else "Low"
    normalized_media_paths = media_paths if media_paths is not None else photo
    normalized_photo = photo if photo is not None else media_paths

    complaint_uid = _generate_unique_complaint_uid(conn)
    cursor.execute(
        """
        INSERT INTO complaints (
            complaint_uid, user_id, location, latitude, longitude, location_key, text,
            category, status, priority, photo, media_paths
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            complaint_uid,
            user_id,
            location,
            latitude,
            longitude,
            location_key or "",
            text,
            category,
            status,
            sanitized_priority,
            normalized_photo,
            normalized_media_paths,
        ),
    )
    complaint_id = cursor.lastrowid

    department_name = CATEGORY_TO_DEPARTMENT.get(category, "Municipal Department")
    department_row = cursor.execute(
        "SELECT id FROM departments WHERE name = ?",
        (department_name,),
    ).fetchone()

    if department_row is None:
        cursor.execute(
            "INSERT INTO departments (name) VALUES (?)",
            (department_name,),
        )
        department_id = cursor.lastrowid
    else:
        department_id = department_row["id"]

    cursor.execute(
        """
        INSERT INTO assignments (complaint_id, department_id)
        VALUES (?, ?)
        """,
        (complaint_id, department_id),
    )

    conn.commit()
    conn.close()
    return complaint_uid, department_name, int(complaint_id)


def fetch_complaints(category: Optional[str] = None) -> List[sqlite3.Row]:
    """Fetch complaints for admin view with optional category filter."""
    conn = get_connection()
    query = """
        SELECT
            c.id AS complaint_id,
            c.complaint_uid,
            COALESCE(u.name, 'Unknown') AS user_name,
            c.location,
            c.location_key,
            c.text,
            c.category,
            c.status,
            c.priority,
            c.is_escalated,
            c.escalation_reason,
            c.escalation_group,
            c.escalation_count,
            c.escalated_at,
            c.photo,
            c.media_paths,
            c.created_at,
            d.name AS department_name
        FROM complaints c
        LEFT JOIN users u ON c.user_id = u.id
        LEFT JOIN assignments a ON c.id = a.complaint_id
        LEFT JOIN departments d ON a.department_id = d.id
    """
    params: Tuple[str, ...] = ()

    if category:
        query += " WHERE c.category = ?"
        params = (category,)

    query += " ORDER BY (c.priority = 'High') DESC, c.is_escalated DESC, c.created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def fetch_open_complaints_same_area_category(
    location_key: str,
    category: str,
    exclude_id: Optional[int] = None,
) -> List[sqlite3.Row]:
    """Fetch unresolved complaints with same area key + category."""
    conn = get_connection()
    query = """
        SELECT id AS complaint_id, text, priority, status
        FROM complaints
        WHERE location_key = ?
          AND category = ?
          AND status != 'Resolved'
    """
    params: List[object] = [location_key, category]
    if exclude_id is not None:
        query += " AND id != ?"
        params.append(exclude_id)
    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()
    return rows


def set_complaints_high_priority(
    complaint_ids: List[int],
    escalation_reason: str,
    escalation_group: str,
    escalation_count: int,
) -> None:
    """Batch update complaint priority and escalation metadata."""
    if not complaint_ids:
        return
    placeholders = ",".join(["?"] * len(complaint_ids))
    conn = get_connection()
    conn.execute(
        f"""
        UPDATE complaints
        SET priority = 'High',
            is_escalated = 1,
            escalation_reason = ?,
            escalation_group = ?,
            escalation_count = ?,
            escalated_at = CURRENT_TIMESTAMP
        WHERE id IN ({placeholders})
        """,
        (escalation_reason, escalation_group, escalation_count, *complaint_ids),
    )
    conn.commit()
    conn.close()


def fetch_user_complaints(
    user_id: int,
    complaint_uid_query: str = "",
    status: str = "",
    category: str = "",
    department: str = "",
) -> List[sqlite3.Row]:
    """Fetch one citizen's complaints with optional UID/status/category filters."""
    conn = get_connection()
    query = """
        SELECT
            c.id AS complaint_id,
            c.user_id,
            c.complaint_uid,
            c.location,
            c.location_key,
            c.text,
            c.category,
            c.status,
            c.priority,
            c.is_escalated,
            c.escalation_reason,
            c.escalation_group,
            c.escalation_count,
            c.escalated_at,
            c.photo,
            c.media_paths,
            c.created_at,
            d.name AS department_name
        FROM complaints c
        LEFT JOIN assignments a ON c.id = a.complaint_id
        LEFT JOIN departments d ON a.department_id = d.id
        WHERE c.user_id = ?
    """
    params = [user_id]

    trimmed_uid_query = complaint_uid_query.strip().upper()
    if trimmed_uid_query:
        query += " AND UPPER(c.complaint_uid) LIKE ?"
        params.append(f"%{trimmed_uid_query}%")

    trimmed_status = status.strip()
    if trimmed_status:
        query += " AND c.status = ?"
        params.append(trimmed_status)

    trimmed_category = category.strip()
    if trimmed_category:
        query += " AND c.category = ?"
        params.append(trimmed_category)

    trimmed_department = department.strip()
    if trimmed_department:
        query += " AND d.name = ?"
        params.append(trimmed_department)

    query += " ORDER BY c.created_at DESC, c.id DESC"
    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()
    return rows


def count_user_complaints(user_id: int) -> int:
    """Return total complaints submitted by one citizen account."""
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) AS total FROM complaints WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    return int((row["total"] if row else 0) or 0)


def _next_status(current_status: str) -> str:
    """Return next status in predefined status flow."""
    if current_status not in STATUS_FLOW:
        return STATUS_FLOW[0]
    current_index = STATUS_FLOW.index(current_status)
    if current_index >= len(STATUS_FLOW) - 1:
        return STATUS_FLOW[-1]
    return STATUS_FLOW[current_index + 1]


def update_complaint_status(complaint_uid: str) -> Optional[str]:
    """Advance complaint status to next stage and return new status."""
    conn = get_connection()
    row = conn.execute(
        "SELECT status FROM complaints WHERE complaint_uid = ?",
        (complaint_uid,),
    ).fetchone()
    if row is None:
        conn.close()
        return None

    new_status = _next_status(row["status"])
    resolved_at = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if new_status == "Resolved"
        else None
    )
    conn.execute(
        "UPDATE complaints SET status = ?, resolved_at = ? WHERE complaint_uid = ?",
        (new_status, resolved_at, complaint_uid),
    )
    conn.commit()
    conn.close()
    return new_status


def set_complaint_status_by_id(complaint_id: int, new_status: str) -> Optional[str]:
    """Set complaint status to an explicit valid value."""
    if new_status not in STATUS_FLOW:
        return None

    conn = get_connection()
    row = conn.execute(
        "SELECT id FROM complaints WHERE id = ?",
        (complaint_id,),
    ).fetchone()
    if row is None:
        conn.close()
        return None

    resolved_at = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if new_status == "Resolved"
        else None
    )
    conn.execute(
        "UPDATE complaints SET status = ?, resolved_at = ? WHERE id = ?",
        (new_status, resolved_at, complaint_id),
    )
    conn.commit()
    conn.close()
    return new_status


def get_complaint_notification_payload(complaint_id: int) -> Optional[sqlite3.Row]:
    """Return complaint + user + department details for notification emails."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT
            c.id AS complaint_id,
            c.complaint_uid,
            c.user_id,
            COALESCE(u.name, u.username, 'Citizen') AS user_name,
            u.email AS user_email,
            c.location,
            c.text,
            c.category,
            c.status,
            c.resolved_at,
            c.resolution_email_sent_at,
            d.name AS department_name
        FROM complaints c
        LEFT JOIN users u ON c.user_id = u.id
        LEFT JOIN assignments a ON c.id = a.complaint_id
        LEFT JOIN departments d ON a.department_id = d.id
        WHERE c.id = ?
        """,
        (complaint_id,),
    ).fetchone()
    conn.close()
    return row


def set_resolution_email_sent_at(complaint_id: int, sent_at: str) -> None:
    """Store timestamp for resolution email delivery."""
    conn = get_connection()
    conn.execute(
        "UPDATE complaints SET resolution_email_sent_at = ? WHERE id = ?",
        (sent_at, complaint_id),
    )
    conn.commit()
    conn.close()


def get_complaint_by_id(complaint_id: int) -> Optional[sqlite3.Row]:
    """Get one complaint row by internal complaint id."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT
            c.id AS complaint_id,
            c.category
        FROM complaints c
        WHERE c.id = ?
        """,
        (complaint_id,),
    ).fetchone()
    conn.close()
    return row


def get_complaint_by_uid(complaint_uid: str) -> Optional[sqlite3.Row]:
    """Get one complaint by public tracking ID."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT
            c.id AS complaint_id,
            c.user_id,
            c.complaint_uid,
            u.name AS user_name,
            c.location,
            c.location_key,
            c.text,
            c.category,
            c.status,
            c.priority,
            c.is_escalated,
            c.escalation_reason,
            c.escalation_group,
            c.escalation_count,
            c.escalated_at,
            c.photo,
            c.media_paths,
            c.created_at,
            d.name AS department_name
        FROM complaints c
        JOIN users u ON c.user_id = u.id
        LEFT JOIN assignments a ON c.id = a.complaint_id
        LEFT JOIN departments d ON a.department_id = d.id
        WHERE c.complaint_uid = ?
        """,
        (complaint_uid,),
    ).fetchone()
    conn.close()
    return row


def get_feedback_by_complaint_uid(complaint_uid: str) -> Optional[sqlite3.Row]:
    """Return feedback row for a complaint UID if it exists."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT id, complaint_uid, rating, message, created_at
        FROM complaint_feedback
        WHERE complaint_uid = ?
        """,
        (complaint_uid,),
    ).fetchone()
    conn.close()
    return row


def record_complaint_feedback(
    complaint_uid: str,
    rating: Optional[int],
    message: str,
) -> Tuple[bool, str]:
    """Create one feedback entry per complaint uid."""
    conn = get_connection()
    cursor = conn.cursor()

    complaint_row = cursor.execute(
        "SELECT id, user_id FROM complaints WHERE complaint_uid = ?",
        (complaint_uid,),
    ).fetchone()
    if complaint_row is None:
        conn.close()
        return False, "Complaint not found."

    existing = cursor.execute(
        "SELECT id FROM complaint_feedback WHERE complaint_id = ?",
        (complaint_row["id"],),
    ).fetchone()
    if existing is not None:
        conn.close()
        return False, "Feedback already submitted."

    cursor.execute(
        """
        INSERT INTO complaint_feedback (complaint_id, complaint_uid, user_id, rating, message)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            complaint_row["id"],
            complaint_uid,
            complaint_row["user_id"],
            rating,
            message,
        ),
    )
    conn.commit()
    conn.close()
    return True, "Feedback submitted."


def get_analytics_data(category: Optional[str] = None) -> Dict[str, object]:
    """Return aggregated counts for analytics dashboard charts."""
    conn = get_connection()
    where_clause = ""
    params: Tuple[str, ...] = ()
    if category:
        where_clause = " WHERE category = ?"
        params = (category,)

    category_labels = list(CATEGORY_TO_DEPARTMENT.keys())
    category_counts_map = {label: 0 for label in category_labels}
    for row in conn.execute(
        f"SELECT category, COUNT(*) AS total FROM complaints{where_clause} GROUP BY category",
        params,
    ).fetchall():
        category_counts_map[row["category"]] = row["total"]

    status_labels = STATUS_FLOW.copy()
    status_counts_map = {label: 0 for label in status_labels}
    for row in conn.execute(
        f"SELECT status, COUNT(*) AS total FROM complaints{where_clause} GROUP BY status",
        params,
    ).fetchall():
        status_counts_map[row["status"]] = row["total"]

    total_complaints = conn.execute(
        f"SELECT COUNT(*) AS total FROM complaints{where_clause}",
        params,
    ).fetchone()["total"]

    total_this_month_query = """
        SELECT COUNT(*) AS total
        FROM complaints
        WHERE strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now', 'localtime')
    """
    if category:
        total_this_month_query += " AND category = ?"
        total_this_month = conn.execute(total_this_month_query, params).fetchone()["total"]
    else:
        total_this_month = conn.execute(total_this_month_query).fetchone()["total"]

    avg_resolution_query = """
        SELECT AVG(julianday(resolved_at) - julianday(created_at)) AS avg_days
        FROM complaints
        WHERE status = 'Resolved' AND resolved_at IS NOT NULL
    """
    if category:
        avg_resolution_query += " AND category = ?"
        avg_days_row = conn.execute(avg_resolution_query, params).fetchone()
    else:
        avg_days_row = conn.execute(avg_resolution_query).fetchone()
    avg_resolution_days = float(avg_days_row["avg_days"] or 0.0)

    received_query = """
        SELECT date(created_at) AS day, COUNT(*) AS total
        FROM complaints
        WHERE date(created_at) >= date('now', 'localtime', '-6 days')
    """
    if category:
        received_query += " AND category = ?"
        received_rows = conn.execute(received_query + " GROUP BY day", params).fetchall()
    else:
        received_rows = conn.execute(received_query + " GROUP BY day").fetchall()
    received_map = {row["day"]: row["total"] for row in received_rows}

    resolved_query = """
        SELECT date(COALESCE(resolved_at, created_at)) AS day, COUNT(*) AS total
        FROM complaints
        WHERE status = 'Resolved'
          AND date(COALESCE(resolved_at, created_at)) >= date('now', 'localtime', '-6 days')
    """
    if category:
        resolved_query += " AND category = ?"
        resolved_rows = conn.execute(resolved_query + " GROUP BY day", params).fetchall()
    else:
        resolved_rows = conn.execute(resolved_query + " GROUP BY day").fetchall()
    resolved_map = {row["day"]: row["total"] for row in resolved_rows}

    area_scope_clause = "WHERE status != 'Resolved'"
    area_params: Tuple[str, ...] = ()
    if category:
        area_scope_clause += " AND category = ?"
        area_params = (category,)

    top_area_rows = conn.execute(
        f"""
        SELECT location_key, COUNT(*) AS total
        FROM complaints
        {area_scope_clause}
        GROUP BY location_key
        ORDER BY total DESC
        LIMIT 5
        """,
        area_params,
    ).fetchall()
    top_area_labels = [row["location_key"] or "Unknown" for row in top_area_rows]
    top_area_counts = [row["total"] for row in top_area_rows]

    repeated_category_rows = conn.execute(
        f"""
        SELECT category, COUNT(*) AS total
        FROM complaints
        {area_scope_clause}
        GROUP BY category
        ORDER BY total DESC
        LIMIT 5
        """,
        area_params,
    ).fetchall()
    repeated_category_labels = [row["category"] for row in repeated_category_rows]
    repeated_category_counts = [row["total"] for row in repeated_category_rows]

    escalated_rows = conn.execute(
        f"""
        SELECT location_key, COUNT(*) AS total
        FROM complaints
        WHERE is_escalated = 1
        {"AND category = ?" if category else ""}
        GROUP BY location_key
        ORDER BY total DESC
        LIMIT 5
        """,
        params,
    ).fetchall()
    escalated_area_labels = [row["location_key"] or "Unknown" for row in escalated_rows]
    escalated_area_counts = [row["total"] for row in escalated_rows]

    conn.close()

    most_common_issue = None
    if total_complaints > 0:
        most_common_issue = max(
            category_counts_map.items(),
            key=lambda item: item[1],
        )[0]

    pending_total = status_counts_map.get("Pending", 0)
    resolved_total = status_counts_map.get("Resolved", 0)
    in_progress_total = status_counts_map.get("In Progress", 0)
    resolution_rate = (resolved_total / total_complaints * 100) if total_complaints else 0.0
    pending_rate = (pending_total / total_complaints * 100) if total_complaints else 0.0

    week_dates = [
        datetime.now().date() - timedelta(days=offset)
        for offset in range(6, -1, -1)
    ]
    weekly_labels = [day.strftime("%a") for day in week_dates]
    weekly_keys = [day.isoformat() for day in week_dates]
    weekly_received = [received_map.get(key, 0) for key in weekly_keys]
    weekly_resolved = [resolved_map.get(key, 0) for key in weekly_keys]

    return {
        "total_complaints": total_complaints,
        "category_labels": category_labels,
        "category_counts": [category_counts_map[label] for label in category_labels],
        "status_labels": status_labels,
        "status_counts": [status_counts_map[label] for label in status_labels],
        "most_common_issue": most_common_issue or "N/A",
        "pending_total": pending_total,
        "in_progress_total": in_progress_total,
        "resolved_total": resolved_total,
        "avg_resolution_days": round(avg_resolution_days, 1),
        "resolution_rate": round(resolution_rate, 1),
        "pending_rate": round(pending_rate, 1),
        "total_this_month": total_this_month,
        "weekly_labels": weekly_labels,
        "weekly_received": weekly_received,
        "weekly_resolved": weekly_resolved,
        "top_area_labels": top_area_labels,
        "top_area_counts": top_area_counts,
        "repeated_category_labels": repeated_category_labels,
        "repeated_category_counts": repeated_category_counts,
        "escalated_area_labels": escalated_area_labels,
        "escalated_area_counts": escalated_area_counts,
    }
