"""
chatbot.py
Upgraded Inference Engine, Multi-Turn Context Manager, and Safety Orchestrator
for Mindora Mental Health Support Chatbot.
STRICT CONSTRAINT: ONLY NumPy and standard Python libraries.
"""

import os
import random
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from preprocess import Preprocessor, clean_text
from model import NumPyLSTM
from conversation import SessionManager, ConversationState
from dialogue_manager import (
    DialogueManager,
    check_explicit_crisis,
    CRISIS_RESPONSE,
    generate_conversation_title
)

DEFAULT_CONFIDENCE_THRESHOLD = 0.45


class MentalHealthChatbot:
    """
    Mental Health Support Chatbot orchestrating:
    1. Independent Pre-Inference Crisis Safety Layer (Explicit vs Ambiguous separation)
    2. NumPy-based LSTM inference (Word-level intent classification)
    3. Multi-Turn Dialogue Management & Session History Tracking
    4. Empathetic Contextual Response Generation with Non-Repetitive Inquiries
    """
    def __init__(
        self,
        vocab_path: str = "data/vocabulary.json",
        tags_path: str = "data/tags.json",
        weights_path: str = "models/lstm_weights.npz",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ):
        self.vocab_path = vocab_path
        self.tags_path = tags_path
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold

        self.preprocessor = Preprocessor()
        self.model: Optional[NumPyLSTM] = None
        self.is_ready = False

        self.session_manager = SessionManager()
        self.dialogue_manager = DialogueManager(confidence_threshold=confidence_threshold)

        self._load_components()

    def _load_components(self):
        """Loads vocabulary, tags, and trained NumPy weights."""
        if not (os.path.exists(self.vocab_path) and os.path.exists(self.tags_path)):
            print(f"[Chatbot] Metadata files missing ({self.vocab_path}, {self.tags_path}). Model not loaded.")
            return

        self.preprocessor.load_metadata(self.vocab_path, self.tags_path)

        if not os.path.exists(self.weights_path):
            print(f"[Chatbot] Weights file missing ({self.weights_path}). Model not loaded.")
            return

        self.model = NumPyLSTM.load_weights(self.weights_path)
        self.is_ready = True
        print(f"[Chatbot] Model and preprocessor loaded successfully! Ready for inference.")

    def check_crisis(self, message: str) -> bool:
        """
        Public safety helper: returns True ONLY if the message contains
        unambiguous, explicit suicidal or active self-harm expressions.
        """
        return check_explicit_crisis(message)

    def get_response(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        history_messages: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generates an empathetic, context-aware chatbot response across multiple turns.
        
        Args:
            user_message: Raw user input text
            session_id: Optional unique session tracking token
            history_messages: Optional list of past messages loaded from the database
            
        Returns:
            dict with {response, intent, confidence, is_crisis, session_id}
        """
        raw_msg = user_message.strip() if user_message else ""

        # Retrieve state from persistent history or in-memory cache
        if history_messages and session_id:
            state = ConversationState.from_db_messages(session_id, history_messages)
            self.session_manager.set_session(session_id, state)
        else:
            state = self.session_manager.get_or_create(session_id)

        # 1. Handle empty input
        if not raw_msg:
            return {
                "response": "I'm listening. Please feel free to share whatever is on your mind whenever you feel ready.",
                "intent": "empty_input",
                "confidence": 1.0,
                "is_crisis": False,
                "session_id": state.session_id
            }

        # 2. Record user message into multi-turn conversation state
        state.add_user_message(raw_msg)

        # 3. Independent Pre-Inference Crisis Safety Layer
        # ONLY triggers emergency screen if statement is explicitly suicidal or self-harming
        if self.check_crisis(raw_msg):
            state.add_bot_response(CRISIS_RESPONSE, "crisis_intervention", 1.0, is_crisis=True)
            return {
                "response": CRISIS_RESPONSE,
                "intent": "crisis_intervention",
                "confidence": 1.0,
                "is_crisis": True,
                "session_id": state.session_id
            }

        # 4. Model availability check
        if not self.is_ready or self.model is None:
            return {
                "response": "The chatbot model is currently initializing. Please try again in a few moments.",
                "intent": "system_unready",
                "confidence": 0.0,
                "is_crisis": False,
                "session_id": state.session_id
            }

        # 5. Preprocess input message
        seq, actual_len = self.preprocessor.text_to_sequence(raw_msg)
        X = np.array([seq], dtype=np.int32)
        lengths = np.array([actual_len], dtype=np.int32)

        # 6. Predict intent via NumPy LSTM forward pass
        preds, confidences = self.model.predict(X, lengths)
        top_idx = int(preds[0])
        model_confidence = float(confidences[0])
        predicted_tag = self.preprocessor.idx2tag.get(top_idx, "unknown")

        # Retrieve available dataset responses for this intent
        dataset_responses = self.preprocessor.responses_by_tag.get(predicted_tag, [])

        # 7. Evaluate multi-turn dialogue context & generate response
        response_text, resolved_intent, resolved_confidence, is_crisis = (
            self.dialogue_manager.evaluate_multi_turn_context(
                user_message=raw_msg,
                state=state,
                predicted_tag=predicted_tag,
                confidence=model_confidence,
                dataset_responses=dataset_responses
            )
        )

        # 8. Record bot response into state history
        state.add_bot_response(response_text, resolved_intent, resolved_confidence, is_crisis)

        return {
            "response": response_text,
            "intent": resolved_intent,
            "confidence": round(resolved_confidence, 4),
            "is_crisis": is_crisis,
            "session_id": state.session_id
        }
