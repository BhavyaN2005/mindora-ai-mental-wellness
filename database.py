"""
database.py
Persistent SQLite Database Engine for Mindora Mental Wellness Companion.
Manages Users, Conversations, Messages, and Session Isolation.
STRICT CONSTRAINT: Pure Python standard library (sqlite3, uuid, datetime).
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from werkzeug.security import generate_password_hash, check_password_hash


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Returns a SQLite connection with foreign keys enabled and row factory set to dict."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str):
    """Initializes tables and indexes if they do not exist."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()

        # 1. Users Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL COLLATE NOCASE,
            email TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            theme_pref TEXT NOT NULL DEFAULT 'system',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # 2. Conversations Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id INTEGER NULL,
            guest_id TEXT NULL,
            title TEXT NOT NULL DEFAULT 'New Conversation',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """)

        # 3. Messages Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL, -- 'user' or 'assistant'
            content TEXT NOT NULL,
            intent TEXT NULL,
            confidence REAL NULL,
            is_crisis INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        """)

        # Indexes for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_guest ON conversations(guest_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, created_at ASC);")
        
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# USER MANAGEMENT & AUTHENTICATION
# --------------------------------------------------------------------------

def create_user(db_path: str, username: str, email: str, password: str) -> Dict[str, Any]:
    """Creates a new user with securely hashed password."""
    username = username.strip()
    email = email.strip().lower()

    if not username or len(username) < 2:
        raise ValueError("Username must be at least 2 characters long.")
    if not email or "@" not in email:
        raise ValueError("A valid email address is required.")
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters long.")

    password_hash = generate_password_hash(password)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )
        user_id = cursor.lastrowid
        conn.commit()
        return {
            "id": user_id,
            "username": username,
            "email": email,
            "theme_pref": "system",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    except sqlite3.IntegrityError as e:
        err_msg = str(e).lower()
        if "username" in err_msg:
            raise ValueError("This username is already taken. Please choose another.")
        elif "email" in err_msg:
            raise ValueError("An account with this email already exists.")
        raise ValueError("Registration failed. Please check your details.")
    finally:
        conn.close()


def authenticate_user(db_path: str, login_identifier: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticates a user via username or email."""
    ident = login_identifier.strip()
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, email, password_hash, theme_pref, created_at FROM users WHERE username = ? OR email = ?",
            (ident, ident.lower())
        )
        row = cursor.fetchone()
        if not row:
            return None

        if check_password_hash(row["password_hash"], password):
            return {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "theme_pref": row["theme_pref"],
                "created_at": row["created_at"]
            }
        return None
    finally:
        conn.close()


def get_user_by_id(db_path: str, user_id: int) -> Optional[Dict[str, Any]]:
    """Fetches user details by primary ID."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, email, theme_pref, created_at FROM users WHERE id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_user_theme(db_path: str, user_id: int, theme_pref: str):
    """Updates user theme preference ('light', 'dark', 'system')."""
    if theme_pref not in ("light", "dark", "system"):
        theme_pref = "system"
    conn = get_db_connection(db_path)
    try:
        conn.execute("UPDATE users SET theme_pref = ? WHERE id = ?", (theme_pref, user_id))
        conn.commit()
    finally:
        conn.close()


def delete_user_account(db_path: str, user_id: int):
    """Deletes user account and cascades deletion to all associated conversations."""
    conn = get_db_connection(db_path)
    try:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# CONVERSATION MANAGEMENT
# --------------------------------------------------------------------------

def create_conversation(
    db_path: str,
    user_id: Optional[int] = None,
    guest_id: Optional[str] = None,
    title: str = "New Conversation"
) -> Dict[str, Any]:
    """Creates a fresh conversation container for an authenticated user or guest."""
    conv_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO conversations (id, user_id, guest_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (conv_id, user_id, guest_id, title, now, now)
        )
        conn.commit()
        return {
            "id": conv_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "last_message": ""
        }
    finally:
        conn.close()


def list_conversations(
    db_path: str,
    user_id: Optional[int] = None,
    guest_id: Optional[str] = None,
    search_query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Lists conversations owned by the user or guest, ordered by updated_at descending.
    Includes message count and preview snippet of the last message.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        query = """
        SELECT 
            c.id, 
            c.title, 
            c.created_at, 
            c.updated_at,
            COUNT(m.id) as message_count,
            (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1) as last_message
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_id
        WHERE 
        """
        params: List[Any] = []
        if user_id is not None:
            query += "c.user_id = ?"
            params.append(user_id)
        elif guest_id is not None:
            query += "c.guest_id = ?"
            params.append(guest_id)
        else:
            return []

        if search_query and search_query.strip():
            query += " AND (c.title LIKE ? OR c.id IN (SELECT conversation_id FROM messages WHERE content LIKE ?))"
            like_term = f"%{search_query.strip()}%"
            params.extend([like_term, like_term])

        query += " GROUP BY c.id ORDER BY c.updated_at DESC"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_conversation(
    db_path: str,
    conv_id: str,
    user_id: Optional[int] = None,
    guest_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Fetches conversation metadata and checks ownership."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        if user_id is not None:
            cursor.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?",
                (conv_id, user_id)
            )
        elif guest_id is not None:
            cursor.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND guest_id = ?",
                (conv_id, guest_id)
            )
        else:
            return None
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_conversation_title(
    db_path: str,
    conv_id: str,
    title: str,
    user_id: Optional[int] = None,
    guest_id: Optional[str] = None
) -> bool:
    """Updates a conversation's title with ownership verification."""
    clean_title = title.strip()
    if not clean_title:
        clean_title = "Conversation"

    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        if user_id is not None:
            cursor.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (clean_title, now, conv_id, user_id)
            )
        elif guest_id is not None:
            cursor.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND guest_id = ?",
                (clean_title, now, conv_id, guest_id)
            )
        else:
            return False
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def delete_conversation(
    db_path: str,
    conv_id: str,
    user_id: Optional[int] = None,
    guest_id: Optional[str] = None
) -> bool:
    """Deletes a single conversation and all its messages."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        if user_id is not None:
            cursor.execute("DELETE FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id))
        elif guest_id is not None:
            cursor.execute("DELETE FROM conversations WHERE id = ? AND guest_id = ?", (conv_id, guest_id))
        else:
            return False
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def clear_all_conversations(
    db_path: str,
    user_id: Optional[int] = None,
    guest_id: Optional[str] = None
):
    """Clears all conversations for the given user or guest."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        if user_id is not None:
            cursor.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
        elif guest_id is not None:
            cursor.execute("DELETE FROM conversations WHERE guest_id = ?", (guest_id,))
        conn.commit()
    finally:
        conn.close()


def link_guest_conversations_to_user(db_path: str, guest_id: str, user_id: int):
    """Associates existing guest conversations with a newly registered/logged-in user."""
    if not guest_id or not user_id:
        return
    conn = get_db_connection(db_path)
    try:
        conn.execute(
            "UPDATE conversations SET user_id = ?, guest_id = NULL WHERE guest_id = ?",
            (user_id, guest_id)
        )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# MESSAGE MANAGEMENT
# --------------------------------------------------------------------------

def add_message(
    db_path: str,
    conv_id: str,
    role: str,
    content: str,
    intent: Optional[str] = None,
    confidence: Optional[float] = None,
    is_crisis: bool = False
) -> Dict[str, Any]:
    """Appends a message to a conversation and updates the conversation timestamp."""
    msg_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, intent, confidence, is_crisis, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (msg_id, conv_id, role, content, intent, confidence, 1 if is_crisis else 0, now)
        )
        # Touch conversation updated_at
        cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
        conn.commit()
        return {
            "id": msg_id,
            "conversation_id": conv_id,
            "role": role,
            "content": content,
            "intent": intent,
            "confidence": confidence,
            "is_crisis": bool(is_crisis),
            "created_at": now
        }
    finally:
        conn.close()


def get_messages(db_path: str, conv_id: str) -> List[Dict[str, Any]]:
    """Fetches all messages in chronological order for a conversation."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, conversation_id, role, content, intent, confidence, is_crisis, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            """,
            (conv_id,)
        )
        rows = cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["is_crisis"] = bool(d["is_crisis"])
            result.append(d)
        return result
    finally:
        conn.close()
