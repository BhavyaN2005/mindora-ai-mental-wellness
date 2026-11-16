"""
dialogue_manager.py
Multi-turn empathetic dialogue orchestration, context resolution, and safety filtering.
STRICT CONSTRAINT: Pure Python standard library.
"""

import re
import random
from typing import Dict, Any, Tuple, Optional, List
from conversation import ConversationState

# --------------------------------------------------------------------------
# 1. CRISIS SAFETY CLASSIFICATION RULES
# --------------------------------------------------------------------------

# Explicit statements indicating suicidal intent, active self-harm, or immediate danger
EXPLICIT_CRISIS_PATTERNS = [
    r"\bkill\s+myself\b",
    r"\bcommit(ting)?\s+suicide\b",
    r"\bwant\s+to\s+die\b",
    r"\bwish\s+i\s+(were|was)\s+dead\b",
    r"\bwant\s+to\s+be\s+dead\b",
    r"\bwant\s+to\s+end\s+my\s+life\b",
    r"\bgoing\s+to\s+kill\s+myself\b",
    r"\bgoing\s+to\s+end\s+my\s+life\b",
    r"\btake\s+my\s+(own\s+)?life\b",
    r"\bplan\s+to\s+kill\s+myself\b",
    r"\bhang\s+myself\b",
    r"\bslit\s+my\s+wrists?\b",
    r"\bcut\s+myself\b",
    r"\bcutting\s+myself\b",
    r"\bharm(ing)?\s+myself\b",
    r"\bhurt(ing)?\s+myself\b",
    r"\boverdose\s+on\b",
    r"\bshoot\s+myself\b",
    r"\bbetter\s+off\s+dead\b",
    r"\bdon't\s+want\s+to\s+(be\s+alive|live)\b",
    r"\bcan't\s+go\s+on\s+living\b",
    r"\bcannot\s+go\s+on\s+living\b",
    r"\bsuicid(e|al)\b"
]

# Ambiguous distress expressions requiring a gentle clarifying question (NOT emergency lockout)
AMBIGUOUS_DISTRESS_PATTERNS = [
    r"\btired\s+of\s+everything\b",
    r"\btired\s+of\s+life\b",
    r"\btired\s+of\s+living\b",
    r"\bwhat('s|\s+is)\s+the\s+point\s+of\s+(living|anything|life)\b",
    r"\bwant\s+it\s+all\s+to\s+stop\b",
    r"\bend\s+it\s+all\b"
]

# Explicit crisis emergency response with verified lifelines
CRISIS_RESPONSE = (
    "It sounds like you are going through an extraordinarily difficult time right now, "
    "and I want you to know that your life has tremendous value and you do not have to face this alone.\n\n"
    "**Immediate Help is Available 24/7 (Free and Confidential):**\n"
    "• **US & Canada:** Call or text **988**, or chat at [988lifeline.org](https://988lifeline.org)\n"
    "• **United Kingdom:** Call **111** (NHS Mental Health) or **999** for emergencies, or call **116 123** (Samaritans)\n"
    "• **India:** Call **112** (Emergency), **14416** (Tele-MANAS), or **9152987821** (iCall)\n"
    "• **Australia:** Call **13 11 14** (Lifeline)\n"
    "• **Worldwide:** Find verified crisis resources in your country at [findahelpline.com](https://findahelpline.com) or [befrienders.org](https://www.befrienders.org)\n\n"
    "Please reach out right now to one of these services, contact a trusted family member or friend, "
    "or go to the nearest emergency room. I am an automated prototype and cannot provide emergency medical care, "
    "but trained professionals are waiting to support you."
)

AMBIGUOUS_CLARIFICATION_RESPONSE = (
    "Hearing that you're tired of everything sounds like you've been carrying a tremendous amount of weight. "
    "When you say that, do you mean you're feeling deeply drained and overwhelmed with daily life, "
    "or are you having thoughts of wanting to hurt yourself? I care about your safety and I'm right here with you."
)

# --------------------------------------------------------------------------
# 2. CONVERSATION AUTO-TITLING
# --------------------------------------------------------------------------

def generate_conversation_title(first_user_message: str) -> str:
    """
    Synthesizes a clean, meaningful 2-4 word conversation title from the first message.
    """
    if not first_user_message:
        return "Thoughtful Reflection"

    msg = first_user_message.strip().lower()

    if re.search(r"\b(anxious|anxiety|panic)\b", msg):
        return "Navigating Anxiety"
    if re.search(r"\b(stress|stressed|burnout|overwhelm)\b", msg):
        return "Managing Overwhelm"
    if re.search(r"\b(empty|numb|hollow)\b", msg):
        return "Feelings of Emptiness"
    if re.search(r"\b(lonely|loneliness|alone|isolated)\b", msg):
        return "Feelings of Loneliness"
    if re.search(r"\b(not\s+feeling\s+good|difficult\s+day|bad\s+day)\b", msg):
        return "A Difficult Day"
    if re.search(r"\b(college|school|exams?|grades?|university)\b", msg):
        return "Academic Pressure"
    if re.search(r"\b(sleep|insomnia|tired|exhausted)\b", msg):
        return "Rest & Sleep"
    if re.search(r"\b(meditat|breathe|calm|grounding)\b", msg):
        return "Mindfulness & Grounding"
    if re.search(r"\b(sad|sadness|down|unhappy)\b", msg):
        return "Working Through Sadness"
    if re.search(r"\b(relationship|friend|family|partner)\b", msg):
        return "Relationships & Connection"

    # Truncate and clean first words
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", first_user_message).strip()
    words = cleaned.split()
    if len(words) >= 2:
        return " ".join(words[:4]).capitalize()
    return "Thoughtful Reflection"


# --------------------------------------------------------------------------
# 3. SAFETY CHECKS
# --------------------------------------------------------------------------

def check_explicit_crisis(text: str) -> bool:
    """Returns True ONLY if text contains explicit suicide or self-harm keywords."""
    if not text:
        return False
    lowered = text.lower()
    for pattern in EXPLICIT_CRISIS_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return False


def check_ambiguous_distress(text: str) -> bool:
    """Returns True for ambiguous statements that need gentle safety clarification."""
    if not text:
        return False
    lowered = text.lower()
    for pattern in AMBIGUOUS_DISTRESS_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return False


# --------------------------------------------------------------------------
# 4. DIALOGUE MANAGER
# --------------------------------------------------------------------------

class DialogueManager:
    """
    Orchestrates human-like, empathetic multi-turn responses by weaving
    LSTM intent classifications with persistent conversation history.
    """
    def __init__(self, confidence_threshold: float = 0.45):
        self.confidence_threshold = confidence_threshold

    def evaluate_multi_turn_context(
        self,
        user_message: str,
        state: ConversationState,
        predicted_tag: str,
        confidence: float,
        dataset_responses: List[str]
    ) -> Tuple[str, str, float, bool]:
        """
        Determines the most empathetic, context-aware response for the current turn.
        
        Returns:
            (response_text, resolved_intent, resolved_confidence, is_crisis)
        """
        raw = user_message.strip()
        lowered = raw.lower()

        # Step 1: Check pending clarification from previous turn
        if state.pending_clarification == "safety_check":
            state.pending_clarification = None
            if check_explicit_crisis(raw) or any(w in lowered for w in ["yes", "hurting myself", "hurt myself", "die"]):
                return CRISIS_RESPONSE, "crisis_intervention", 1.0, True
            else:
                response = (
                    "Thank you for being open and clarifying that with me. I'm really glad you're safe. "
                    "Feeling that level of exhaustion is still so hard to bear. "
                    "When everything feels like too much, we don't have to tackle the whole picture right now. "
                    "What has felt like the heaviest thing weighing on your shoulders today?"
                )
                return response, "exhaustion_clarified", 0.95, False

        # Step 2: Check for ambiguous distress statements
        if check_ambiguous_distress(raw):
            state.pending_clarification = "safety_check"
            return AMBIGUOUS_CLARIFICATION_RESPONSE, "ambiguous_distress_clarification", 0.90, False

        # Step 3: Multi-Turn Canonical Dialogues

        # A. Canonical Progression Turn 3: "Maybe loneliness" / "loneliness"
        has_lonely_context = (
            any(t in state.context_topics for t in ["emptiness", "sadness", "isolation"])
            or any(re.search(r"\b(empty|sad|difficult|words|feel)\b", h.get("text", "").lower()) for h in state.history)
            or state.turn_count >= 2
        )
        if re.search(r"\b(maybe\s+)?(loneliness|lonely)\b", lowered) and has_lonely_context:
            state.register_topic("loneliness")
            q_key = "lonely_in_crowd_vs_alone"
            if not state.was_question_asked(q_key):
                state.mark_question_asked(q_key)
                response = (
                    "Do you feel lonely even when you're around people, "
                    "or is it more that you don't have someone you can comfortably open up to?"
                )
                return response, "sad", 0.96, False

        # B. Canonical Progression Turn 2: "I don't know. I just feel empty."
        if ("empty" in lowered or "numb" in lowered or "hollow" in lowered) and (
            "don't know" in lowered or "dont know" in lowered or state.current_intent in ["sad", "not_feeling_good"]
        ):
            state.register_topic("emptiness")
            q_key = "difficult_words_emptiness"
            if not state.was_question_asked(q_key):
                state.mark_question_asked(q_key)
                response = (
                    "Sometimes it's difficult to put a feeling into words. That's okay. "
                    "Does it feel more like loneliness, sadness, or being disconnected from everything?"
                )
                return response, "sad", 0.96, False

        # C. Short uncertainty ("I don't know", "not sure", "hard to explain")
        if re.search(r"^(i\s+)?(don't|dont|do\s+not)\s+know\b", lowered) or lowered in ["not sure", "hard to explain", "no reason"]:
            if state.current_intent in ["sad", "depressed", "not_feeling_good", "stressed", "anxious"] or state.turn_count >= 2:
                q_key = "idk_gentle_validation"
                if not state.was_question_asked(q_key):
                    state.mark_question_asked(q_key)
                    response = (
                        "That is completely okay. You don't have to have words for it, and you certainly don't need "
                        "to justify or explain it. Sometimes our feelings are just heavy and quiet without a neat reason. "
                        "How is your body feeling right now—are you feeling physically drained, tense, or just numb?"
                    )
                else:
                    response = (
                        "It's completely fine not to know. We can take things as slowly as you need. "
                        "Just taking a quiet breath together is enough right now."
                    )
                return response, "acknowledging_uncertainty", 0.90, False
            else:
                response = (
                    "That's completely fine. Sometimes it takes time to unravel how we're feeling. "
                    "We don't have to figure anything out right now. I'm just here to listen."
                )
                return response, "acknowledging_uncertainty", 0.85, False

        # D. Emptiness standalone ("I feel empty", "feeling numb")
        if re.search(r"\b(feel\s+)?(empty|numb|hollow|blank)\b", lowered):
            state.register_topic("emptiness")
            q_key = "emptiness_reflection"
            if not state.was_question_asked(q_key):
                state.mark_question_asked(q_key)
                response = (
                    "Sometimes it's difficult to put a feeling into words. That's okay. "
                    "Does it feel more like loneliness, sadness, or being disconnected from everything?"
                )
            else:
                response = (
                    "That feeling of emptiness takes a real toll. Please remember you don't have to force yourself "
                    "to feel cheerful or productive right now. How has your sleep or rest been lately?"
                )
            return response, "sad", 0.95, False

        # E. "I'm not feeling good" / "I feel bad today" / "I'm having a difficult day"
        if re.search(r"\b(not\s+feeling\s+good|difficult\s+day|having\s+a\s+hard\s+time|feel\s+bad)\b", lowered):
            state.register_topic("sadness")
            q_key = "not_feeling_good_inquiry"
            if not state.was_question_asked(q_key):
                state.mark_question_asked(q_key)
                response = (
                    "I'm sorry you're having a difficult day. "
                    "Has something happened recently, or have you been feeling this way for a while?"
                )
            else:
                response = (
                    "I hear how heavy today is feeling. I can't promise everything will become easy immediately, "
                    "but we can take this one step at a time. What would feel most comforting for you right now?"
                )
            return response, "sad", 0.96, False

        # F. "I've been feeling overwhelmed"
        if re.search(r"\b(overwhelmed|overwhelming|too\s+much\s+to\s+handle)\b", lowered):
            state.register_topic("stress")
            q_key = "overwhelmed_inquiry"
            if not state.was_question_asked(q_key):
                state.mark_question_asked(q_key)
                response = (
                    "Feeling overwhelmed can make even simple tasks feel insurmountable. "
                    "What's been taking up the most mental space for you lately?"
                )
            else:
                response = (
                    "When overwhelm peaks, taking a step back to breathe is essential. "
                    "You don't have to carry the whole load at once. Could you pause for a brief moment?"
                )
            return response, "stressed", 0.94, False

        # G. "I just need someone to listen"
        if re.search(r"\b(need\s+someone\s+to\s+listen|just\s+listen|want\s+to\s+vent)\b", lowered):
            state.register_topic("connection")
            response = (
                "I am right here with you, and there is no pressure to have anything figured out or explained cleanly. "
                "Please take all the space and time you need—what's on your heart?"
            )
            return response, "listening", 0.96, False

        # H. "I don't know how I feel"
        if re.search(r"\b(don't|dont)\s+know\s+how\s+i\s+feel\b", lowered):
            state.register_topic("uncertainty")
            response = (
                "It is completely normal for feelings to feel tangled or blurry. "
                "Sometimes our body notices before our mind does. Are you feeling physically exhausted, tense, or more of an emotional numbness?"
            )
            return response, "acknowledging_uncertainty", 0.92, False

        # I. College / School / Academic Pressure
        if re.search(r"\b(college|university|school|exam|exams|grades?|coursework|homework|professors?|assignments?)\b", lowered):
            state.register_topic("academics")
            q_key = "academics_pressure"
            if not state.was_question_asked(q_key):
                state.mark_question_asked(q_key)
                response = (
                    "College can feel relentless when coursework, deadlines, and expectations start piling up all at once. "
                    "It takes a tremendous amount of mental energy to keep going under that kind of pressure. "
                    "Is it the sheer volume of work, a specific subject, or feeling like you have no time to breathe that feels hardest right now?"
                )
            else:
                response = (
                    "Academic demands can be so draining. When you're dealing with college difficulties, "
                    "it's easy for your own well-being to get pushed to the side. Are you managing to take any breaks to recharge?"
                )
            return response, "stressed", 0.92, False

        # J. Loneliness / Isolation ("I don't have anyone to talk to")
        if re.search(r"\b(don't|dont|have\s+no|nobody|no\s+one)\s+(have\s+anyone|to\s+talk\s+to|one\s+cares|friends?)\b", lowered) or \
           re.search(r"\b(all\s+alone|feeling\s+alone|so\s+isolated)\b", lowered):
            state.register_topic("isolation")
            q_key = "isolation_reflection"
            if not state.was_question_asked(q_key):
                state.mark_question_asked(q_key)
                response = (
                    "Feeling like you have no one you can turn to makes every struggle twice as heavy. "
                    "That kind of silence can feel very loud and isolating. Even though I'm an AI companion, "
                    "I am right here with you, listening without any judgment. "
                    "How long have you felt like you've been carrying everything on your own?"
                )
            else:
                response = (
                    "Carrying that loneliness day after day is deeply exhausting. "
                    "You deserve to be heard and supported. Have you felt this distance with people for a long time, "
                    "or has it felt worse recently?"
                )
            return response, "sad", 0.95, False

        # Step 4: Contextual Smoothing for LSTM Predictions
        # Never allow model prediction alone to trigger crisis!
        if predicted_tag == "suicide":
            response = (
                "It sounds like you're going through a lot of pain right now. "
                "I'm here to listen and support you. Could you share a bit more about what's feeling hardest at the moment?"
            )
            return response, "emotional_support", 0.65, False

        # Low-confidence fallback
        if confidence < self.confidence_threshold:
            if state.context_topics:
                recent_topic = list(state.context_topics)[-1]
                response = (
                    f"I want to make sure I understand you properly. Earlier you mentioned dealing with {recent_topic}—"
                    "is what you're sharing connected to that, or is there another part of your day you're thinking about?"
                )
            else:
                response = (
                    "I want to make sure I support you properly, but I didn't quite catch your meaning. "
                    "Could you rephrase that, or tell me a bit more about how you're feeling right now?"
                )
            return response, "conversational_fallback", round(confidence, 4), False

        # Confident intent from dataset pool
        if dataset_responses:
            chosen = random.choice(dataset_responses)
            softened = self._humanize_dataset_response(chosen, predicted_tag, state)
            return softened, predicted_tag, round(confidence, 4), False

        return (
            "I hear you, and I appreciate you sharing that with me. What would feel most helpful for us to talk through right now?",
            predicted_tag or "general",
            round(confidence, 4),
            False
        )

    def _humanize_dataset_response(self, text: str, tag: str, state: ConversationState) -> str:
        """
        Replaces stiff or repetitive canned dataset responses with warmer,
        more natural conversational phrasing.
        """
        replacements = {
            "Why do you think you feel this way?": "Has something specific happened recently, or has this feeling been building up for a while?",
            "Tell me why do you think you're feeling this way?": "Would you like to share what feels like it's contributing to this, or is it hard to pinpoint?",
            "What do you think is causing this?": "Has something specific triggered this stress, or does it feel like everything is accumulating?",
            "Don't worry, everything will be fine": "I can't promise everything will feel easy immediately, but we can take this one step at a time.",
            "Don't let the little worries bring you down. What's the worse that can happen?": "Even small worries can feel very heavy when they pile up. What's feeling most present for you right now?"
        }

        for target, repl in replacements.items():
            if target in text:
                text = text.replace(target, repl)

        return text
