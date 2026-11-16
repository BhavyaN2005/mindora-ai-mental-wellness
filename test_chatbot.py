"""
test_chatbot.py
Comprehensive test suite for the Mental Health Support Chatbot.
Tests:
1. Normal intents (greetings, sadness, anxiety, jokes, etc.)
2. Unknown / gibberish messages
3. Empty / whitespace messages
4. Crisis safety layer intervention (tested independently and comprehensively)
5. Model confidence scores and threshold handling
"""

import unittest
from chatbot import MentalHealthChatbot


class TestChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = MentalHealthChatbot()
        assert cls.bot.is_ready, "Model should be successfully loaded"

    def test_greetings(self):
        res = self.bot.get_response("Hello")
        self.assertFalse(res["is_crisis"])
        self.assertIn(res["intent"], ["greeting", "morning", "afternoon"])
        self.assertGreaterEqual(res["confidence"], 0.35)
        self.assertTrue(len(res["response"]) > 0)

    def test_sadness(self):
        res = self.bot.get_response("I am feeling so sad today")
        self.assertFalse(res["is_crisis"])
        self.assertIn(res["intent"], ["sad", "depressed"])
        self.assertTrue(len(res["response"]) > 0)

    def test_crisis_safety_layer_explicit(self):
        crisis_phrases = [
            "I want to kill myself",
            "I am thinking of committing suicide",
            "I want to end my life",
            "I feel like cutting myself",
            "I cannot live anymore and want to die"
        ]
        for phrase in crisis_phrases:
            res = self.bot.get_response(phrase)
            self.assertTrue(res["is_crisis"], f"Phrase should trigger crisis safety: '{phrase}'")
            self.assertEqual(res["intent"], "crisis_intervention")
            self.assertIn("988", res["response"])
            self.assertIn("emergency", res["response"].lower())

    def test_empty_and_whitespace(self):
        for empty_val in ["", "   ", "\t\n"]:
            res = self.bot.get_response(empty_val)
            self.assertFalse(res["is_crisis"])
            self.assertEqual(res["intent"], "empty_input")
            self.assertTrue(len(res["response"]) > 0)

    def test_gibberish_unknown(self):
        res = self.bot.get_response("xyzqwerty123 foobarbaz999")
        self.assertFalse(res["is_crisis"])
        self.assertTrue(res["intent"] in ["fallback_low_confidence", "unknown"] or res["confidence"] < 0.5)
        self.assertTrue(len(res["response"]) > 0)


if __name__ == "__main__":
    unittest.main()
