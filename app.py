"""
app.py
Production-Style Flask Web Application for Mindora AI Mental Wellness Companion.
STRICT CONSTRAINT: ONLY Flask, SQLite3, NumPy, and Python standard libraries.
"""

import os
import uuid
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify, session

from config import Config
from database import (
    init_db,
    create_user,
    authenticate_user,
    get_user_by_id,
    update_user_theme,
    delete_user_account,
    create_conversation,
    list_conversations,
    get_conversation,
    update_conversation_title,
    delete_conversation,
    clear_all_conversations,
    link_guest_conversations_to_user,
    add_message,
    get_messages
)
from chatbot import MentalHealthChatbot
from dialogue_manager import generate_conversation_title

app = Flask(__name__)
app.config.from_object(Config)

# Initialize persistent SQLite database
print(f"[Mindora Startup] Initializing database at: {Config.DATABASE_PATH}")
init_db(Config.DATABASE_PATH)

# Initialize trained NumPy-based LSTM chatbot engine
print("[Mindora Startup] Loading NumPy LSTM and preprocessing metadata...")
chatbot = MentalHealthChatbot(
    vocab_path=Config.VOCAB_PATH,
    tags_path=Config.TAGS_PATH,
    weights_path=Config.WEIGHTS_PATH,
    confidence_threshold=Config.CONFIDENCE_THRESHOLD
)


def get_db_path():
    """Returns the active database path from Flask config or Config fallback."""
    return app.config.get("DATABASE_PATH", Config.DATABASE_PATH)


def get_current_actor():
    """
    Returns (user_id, guest_id) for the current request.
    Ensures every visitor has an isolated identity (registered user or guest).
    """
    user_id = session.get("user_id")
    guest_id = session.get("guest_id")

    if not user_id and not guest_id:
        guest_id = str(uuid.uuid4())
        session["guest_id"] = guest_id

    return user_id, guest_id


# --------------------------------------------------------------------------
# FRONTEND ROUTE
# --------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Renders the Mindora mental wellness web application."""
    user_id, _ = get_current_actor()
    current_user = get_user_by_id(get_db_path(), user_id) if user_id else None
    return render_template("index.html", current_user=current_user)


# --------------------------------------------------------------------------
# AUTHENTICATION & USER PROFILE APIS
# --------------------------------------------------------------------------

@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    """Registers a new user and links any existing guest conversations."""
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON."}), 400

    data = request.get_json()
    username = data.get("username", "")
    email = data.get("email", "")
    password = data.get("password", "")

    try:
        user = create_user(get_db_path(), username, email, password)
        # Migrate guest conversations to this new account
        guest_id = session.get("guest_id")
        if guest_id:
            link_guest_conversations_to_user(get_db_path(), guest_id, user["id"])
            session.pop("guest_id", None)

        session["user_id"] = user["id"]
        return jsonify({"success": True, "user": user}), 201
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": "An unexpected error occurred during registration."}), 500


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    """Authenticates user and restores account session."""
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON."}), 400

    data = request.get_json()
    login_id = data.get("login", "")
    password = data.get("password", "")

    if not login_id or not password:
        return jsonify({"success": False, "error": "Please provide your username/email and password."}), 400

    user = authenticate_user(get_db_path(), login_id, password)
    if not user:
        return jsonify({"success": False, "error": "Invalid username/email or password."}), 401

    # Link any active guest chats
    guest_id = session.get("guest_id")
    if guest_id:
        link_guest_conversations_to_user(get_db_path(), guest_id, user["id"])
        session.pop("guest_id", None)

    session["user_id"] = user["id"]
    return jsonify({"success": True, "user": user}), 200


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    """Logs out the current user and clears session credentials."""
    session.pop("user_id", None)
    return jsonify({"success": True, "message": "Logged out successfully."}), 200


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    """Returns current user status, identity, and theme preference."""
    user_id, guest_id = get_current_actor()
    if user_id:
        user = get_user_by_id(get_db_path(), user_id)
        if user:
            return jsonify({
                "authenticated": True,
                "user": user,
                "theme_pref": user.get("theme_pref", "system")
            }), 200

    # Guest user response
    return jsonify({
        "authenticated": False,
        "guest_id": guest_id,
        "theme_pref": session.get("theme_pref", "system")
    }), 200


@app.route("/api/auth/settings", methods=["PATCH"])
def auth_settings():
    """Updates user theme and UI preferences."""
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON."}), 400

    data = request.get_json()
    theme_pref = data.get("theme_pref", "system")
    session["theme_pref"] = theme_pref

    user_id, _ = get_current_actor()
    if user_id:
        update_user_theme(get_db_path(), user_id, theme_pref)

    return jsonify({"success": True, "theme_pref": theme_pref}), 200


@app.route("/api/auth/account", methods=["DELETE"])
def auth_delete_account():
    """Permanently deletes the user account and associated chat history."""
    user_id, _ = get_current_actor()
    if not user_id:
        return jsonify({"success": False, "error": "Only registered users can delete accounts."}), 401

    delete_user_account(get_db_path(), user_id)
    session.clear()
    return jsonify({"success": True, "message": "Account and all associated data permanently deleted."}), 200


# --------------------------------------------------------------------------
# CONVERSATION MANAGEMENT APIS
# --------------------------------------------------------------------------

@app.route("/api/conversations", methods=["GET"])
def get_conversations_list():
    """Fetches list of conversations for the active user or guest session."""
    user_id, guest_id = get_current_actor()
    search = request.args.get("search", "")
    conversations = list_conversations(get_db_path(), user_id=user_id, guest_id=guest_id, search_query=search)
    return jsonify({"success": True, "conversations": conversations}), 200


@app.route("/api/conversations", methods=["POST"])
def post_new_conversation():
    """Explicitly initializes a new conversation."""
    user_id, guest_id = get_current_actor()
    data = request.get_json(silent=True) or {}
    title = data.get("title", "New Conversation")
    conv = create_conversation(get_db_path(), user_id=user_id, guest_id=guest_id, title=title)
    return jsonify({"success": True, "conversation": conv}), 201


@app.route("/api/conversations/<conv_id>", methods=["GET"])
def get_single_conversation(conv_id):
    """Retrieves conversation metadata and all historical messages."""
    user_id, guest_id = get_current_actor()
    conv = get_conversation(get_db_path(), conv_id, user_id=user_id, guest_id=guest_id)
    if not conv:
        return jsonify({"success": False, "error": "Conversation not found or unauthorized."}), 404

    messages = get_messages(get_db_path(), conv_id)
    return jsonify({
        "success": True,
        "conversation": conv,
        "messages": messages
    }), 200


@app.route("/api/conversations/<conv_id>", methods=["PATCH"])
def patch_conversation_title(conv_id):
    """Renames an existing conversation."""
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON."}), 400

    data = request.get_json()
    new_title = data.get("title", "").strip()
    if not new_title:
        return jsonify({"success": False, "error": "Title cannot be empty."}), 400

    user_id, guest_id = get_current_actor()
    updated = update_conversation_title(get_db_path(), conv_id, new_title, user_id=user_id, guest_id=guest_id)
    if not updated:
        return jsonify({"success": False, "error": "Conversation not found or unauthorized."}), 404

    return jsonify({"success": True, "title": new_title}), 200


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
def delete_single_conversation(conv_id):
    """Deletes an individual conversation."""
    user_id, guest_id = get_current_actor()
    deleted = delete_conversation(get_db_path(), conv_id, user_id=user_id, guest_id=guest_id)
    if not deleted:
        return jsonify({"success": False, "error": "Conversation not found or unauthorized."}), 404

    return jsonify({"success": True, "message": "Conversation deleted."}), 200


@app.route("/api/conversations/clear", methods=["POST"])
def post_clear_conversations():
    """Clears all conversations belonging to the active user/guest."""
    user_id, guest_id = get_current_actor()
    clear_all_conversations(get_db_path(), user_id=user_id, guest_id=guest_id)
    return jsonify({"success": True, "message": "All conversations cleared."}), 200


# --------------------------------------------------------------------------
# CHAT INFERENCE & MULTI-TURN PIPELINE
# --------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Receives user message, manages conversation creation, persists user message,
    evaluates multi-turn context with NumPy LSTM, and saves assistant response.
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Request body must be valid JSON."}), 400

    payload = request.get_json()
    message = payload.get("message", "").strip()
    conv_id = payload.get("conversation_id")
    user_id, guest_id = get_current_actor()

    # If no conversation_id or conversation doesn't exist, create one
    conv = None
    if conv_id:
        conv = get_conversation(get_db_path(), conv_id, user_id=user_id, guest_id=guest_id)

    is_first_message = False
    if not conv:
        is_first_message = True
        auto_title = generate_conversation_title(message)
        conv = create_conversation(get_db_path(), user_id=user_id, guest_id=guest_id, title=auto_title)
        conv_id = conv["id"]

    # 1. Fetch recent persistent history for context
    past_messages = get_messages(get_db_path(), conv_id)
    if not past_messages and not is_first_message:
        # If conversation existed but had 0 messages, auto-title from this message
        auto_title = generate_conversation_title(message)
        update_conversation_title(get_db_path(), conv_id, auto_title, user_id=user_id, guest_id=guest_id)

    # 2. Persist user message to SQLite
    user_msg_record = add_message(get_db_path(), conv_id, "user", message)

    # 3. Generate empathetic multi-turn response using NumPy LSTM + Dialogue Manager
    result = chatbot.get_response(
        user_message=message,
        session_id=conv_id,
        history_messages=past_messages
    )

    # 4. Persist assistant response to SQLite
    assistant_msg_record = add_message(
        get_db_path(),
        conv_id=conv_id,
        role="assistant",
        content=result["response"],
        intent=result["intent"],
        confidence=result["confidence"],
        is_crisis=result["is_crisis"]
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    return jsonify({
        "success": True,
        "response": result["response"],
        "intent": result["intent"],
        "confidence": result["confidence"],
        "is_crisis": result["is_crisis"],
        "conversation_id": conv_id,
        "message_id": assistant_msg_record["id"],
        "timestamp": timestamp
    }), 200


# --------------------------------------------------------------------------
# SYSTEM HEALTH ENDPOINT
# --------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    """Production health check confirming model and database readiness."""
    return jsonify({
        "status": "healthy" if chatbot.is_ready else "degraded",
        "model_loaded": chatbot.is_ready,
        "database": "connected",
        "num_classes": chatbot.preprocessor.num_classes if chatbot.is_ready else 0,
        "vocab_size": chatbot.preprocessor.vocab_size if chatbot.is_ready else 0,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Endpoint not found."}), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(e):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Internal server error."}), 500
    return "An internal server error occurred.", 500


if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
