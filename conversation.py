"""
conversation.py
Session-based and database-backed conversation state management for Mindora.
Tracks user turns, emotional trajectory, topic entities, and conversation history.
STRICT CONSTRAINT: Pure Python standard library.
"""

import time
import uuid
import threading
from typing import Dict, List, Optional, Set, Any


class ConversationState:
    """
    Maintains the state of an ongoing multi-turn conversation session.
    Can be initialized fresh or reconstructed from persistent database messages.
    """
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.history: List[Dict[str, Any]] = []
        self.turn_count: int = 0
        self.current_intent: Optional[str] = None
        self.previous_intent: Optional[str] = None
        self.emotional_trajectory: List[str] = []
        self.context_topics: Set[str] = set()
        self.asked_questions: List[str] = []
        self.pending_clarification: Optional[str] = None
        self.last_user_message: str = ""
        self.last_bot_message: str = ""
        self.created_at: float = time.time()
        self.last_active: float = time.time()

    def add_user_message(self, text: str):
        """Records a user turn and updates timestamps."""
        self.turn_count += 1
        self.last_user_message = text
        self.last_active = time.time()
        self.history.append({
            "role": "user",
            "text": text,
            "timestamp": self.last_active,
            "turn": self.turn_count
        })

    def add_bot_response(self, text: str, intent: str, confidence: float, is_crisis: bool = False):
        """Records a bot turn, updates previous intents and history."""
        self.previous_intent = self.current_intent
        self.current_intent = intent
        self.last_bot_message = text
        self.last_active = time.time()
        
        if intent and intent not in ("empty_input", "fallback_low_confidence", "system_unready"):
            self.emotional_trajectory.append(intent)

        self.history.append({
            "role": "bot",
            "text": text,
            "intent": intent,
            "confidence": confidence,
            "is_crisis": is_crisis,
            "timestamp": self.last_active,
            "turn": self.turn_count
        })

    def register_topic(self, topic: str):
        """Adds a recognized context topic (e.g., 'college', 'isolation')."""
        self.context_topics.add(topic)

    def was_question_asked(self, question_key: str) -> bool:
        """Checks if a similar question has already been asked in this session."""
        return question_key in self.asked_questions

    def mark_question_asked(self, question_key: str):
        """Marks a question key as asked to prevent repetitive interrogations."""
        self.asked_questions.append(question_key)

    def get_recent_history(self, limit: int = 8) -> List[Dict[str, Any]]:
        """Returns the most recent N turns of the dialogue."""
        return self.history[-limit:]

    @classmethod
    def from_db_messages(cls, conversation_id: str, db_messages: List[Dict[str, Any]]) -> "ConversationState":
        """
        Reconstructs state seamlessly from persistent database messages.
        Ensures multi-turn awareness survives page reloads and device switches.
        """
        state = cls(conversation_id)
        for msg in db_messages:
            role = msg.get("role")
            content = msg.get("content", "")
            intent = msg.get("intent")
            confidence = msg.get("confidence", 1.0)
            is_crisis = bool(msg.get("is_crisis", False))

            if role == "user":
                state.add_user_message(content)
                c_low = content.lower()
                if "empty" in c_low or "numb" in c_low or "hollow" in c_low:
                    state.register_topic("emptiness")
                if "lonel" in c_low or "alone" in c_low:
                    state.register_topic("isolation")
                if "colleg" in c_low or "exam" in c_low or "school" in c_low:
                    state.register_topic("academics")
                if "stress" in c_low or "overwhelm" in c_low:
                    state.register_topic("stress")
                if "sad" in c_low or "difficult day" in c_low or "not feeling good" in c_low:
                    state.register_topic("sadness")
            elif role == "assistant" or role == "bot":
                state.add_bot_response(content, intent or "general", confidence or 1.0, is_crisis)

        return state


class SessionManager:
    """
    Thread-safe in-memory session cache with automatic TTL expiration.
    """
    def __init__(self, ttl_seconds: int = 7200):
        self._sessions: Dict[str, ConversationState] = {}
        self._lock = threading.Lock()
        self.ttl_seconds = ttl_seconds

    def get_or_create(self, session_id: Optional[str] = None) -> ConversationState:
        """Retrieves an existing session or creates a fresh one."""
        with self._lock:
            self._cleanup_expired()
            if not session_id or session_id not in self._sessions:
                new_id = session_id if (session_id and len(session_id) >= 8) else str(uuid.uuid4())
                state = ConversationState(new_id)
                self._sessions[new_id] = state
                return state
            return self._sessions[session_id]

    def set_session(self, session_id: str, state: ConversationState):
        """Sets or replaces a session state."""
        with self._lock:
            self._sessions[session_id] = state

    def reset_session(self, session_id: str) -> ConversationState:
        """Clears an existing session and starts fresh."""
        with self._lock:
            new_state = ConversationState(session_id)
            self._sessions[session_id] = new_state
            return new_state

    def _cleanup_expired(self):
        """Evicts sessions inactive longer than TTL."""
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s.last_active > self.ttl_seconds]
        for sid in expired:
            del self._sessions[sid]
