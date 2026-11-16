"""
test_app.py
Comprehensive integration test suite for Mindora's Flask backend,
database persistence, authentication, conversation management, and chat APIs.
"""

import unittest
import json
import os
from app import app
from config import Config
from database import init_db


class MindoraAppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.test_db = f"data/test_integration_{os.getpid()}.db"
        app.config["DATABASE_PATH"] = cls.test_db
        if os.path.exists(cls.test_db):
            try:
                os.remove(cls.test_db)
            except Exception:
                pass
        init_db(cls.test_db)
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "test_db") and os.path.exists(cls.test_db):
            try:
                os.remove(cls.test_db)
            except Exception:
                pass

    def test_01_index_view(self):
        """Test GET / renders index.html with Mindora branding."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Mindora", res.data)
        self.assertIn(b"Your space to feel heard", res.data)

    def test_02_health_endpoint(self):
        """Test GET /api/health verifies model and DB readiness."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["database"], "connected")
        self.assertGreater(data["num_classes"], 0)

    def test_03_auth_flow(self):
        """Test user registration, duplicate prevention, login, and logout."""
        # 1. Register
        reg_payload = {
            "username": "mindora_user",
            "email": "user@mindora.org",
            "password": "secure_password_123"
        }
        res_reg = self.client.post("/api/auth/register", json=reg_payload)
        self.assertEqual(res_reg.status_code, 201)
        data_reg = json.loads(res_reg.data)
        self.assertTrue(data_reg["success"])
        self.assertEqual(data_reg["user"]["username"], "mindora_user")

        # 2. Check /api/auth/me
        res_me = self.client.get("/api/auth/me")
        data_me = json.loads(res_me.data)
        self.assertTrue(data_me["authenticated"])
        self.assertEqual(data_me["user"]["username"], "mindora_user")

        # 3. Duplicate register should fail
        res_dup = self.client.post("/api/auth/register", json=reg_payload)
        self.assertEqual(res_dup.status_code, 400)

        # 4. Logout
        res_logout = self.client.post("/api/auth/logout")
        self.assertEqual(res_logout.status_code, 200)

        res_me_guest = self.client.get("/api/auth/me")
        data_me_guest = json.loads(res_me_guest.data)
        self.assertFalse(data_me_guest["authenticated"])

        # 5. Login
        login_payload = {
            "login": "mindora_user",
            "password": "secure_password_123"
        }
        res_login = self.client.post("/api/auth/login", json=login_payload)
        self.assertEqual(res_login.status_code, 200)
        data_login = json.loads(res_login.data)
        self.assertTrue(data_login["success"])

    def test_04_conversations_and_chat_persistence(self):
        """Test creating conversation, chatting, persisting messages, and renaming."""
        # 1. Send first chat message (auto-creates conversation)
        chat_payload = {"message": "I've been feeling overwhelmed lately with my coursework."}
        res_chat = self.client.post("/api/chat", json=chat_payload)
        self.assertEqual(res_chat.status_code, 200)
        data_chat = json.loads(res_chat.data)
        self.assertTrue(data_chat["success"])
        self.assertFalse(data_chat["is_crisis"])
        conv_id = data_chat["conversation_id"]
        self.assertIsNotNone(conv_id)

        # 2. Fetch conversation details & messages
        res_conv = self.client.get(f"/api/conversations/{conv_id}")
        self.assertEqual(res_conv.status_code, 200)
        data_conv = json.loads(res_conv.data)
        self.assertTrue(data_conv["success"])
        # Should contain 2 messages: user and assistant
        self.assertEqual(len(data_conv["messages"]), 2)
        self.assertEqual(data_conv["messages"][0]["role"], "user")
        self.assertEqual(data_conv["messages"][1]["role"], "assistant")

        # 3. Send follow-up in the same conversation
        followup_payload = {
            "message": "College is getting difficult.",
            "conversation_id": conv_id
        }
        res_followup = self.client.post("/api/chat", json=followup_payload)
        self.assertEqual(res_followup.status_code, 200)
        data_followup = json.loads(res_followup.data)
        self.assertEqual(data_followup["conversation_id"], conv_id)

        # Verify messages now count 4
        res_conv_after = self.client.get(f"/api/conversations/{conv_id}")
        data_conv_after = json.loads(res_conv_after.data)
        self.assertEqual(len(data_conv_after["messages"]), 4)

        # 4. Rename conversation
        rename_payload = {"title": "College Pressure Reflections"}
        res_rename = self.client.patch(f"/api/conversations/{conv_id}", json=rename_payload)
        self.assertEqual(res_rename.status_code, 200)

        res_check_rename = self.client.get(f"/api/conversations/{conv_id}")
        data_check = json.loads(res_check_rename.data)
        self.assertEqual(data_check["conversation"]["title"], "College Pressure Reflections")

    def test_05_crisis_intervention_flow(self):
        """Test that explicit suicidal ideation triggers verified emergency intervention."""
        crisis_payload = {"message": "I want to kill myself right now"}
        res = self.client.post("/api/chat", json=crisis_payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data["success"])
        self.assertTrue(data["is_crisis"])
        self.assertEqual(data["intent"], "crisis_intervention")
        self.assertIn("988", data["response"])
        self.assertIn("emergency", data["response"].lower())

    def test_06_delete_and_clear_conversations(self):
        """Test deleting single conversation and clearing all conversations."""
        # Create a new conversation
        res_new = self.client.post("/api/conversations", json={"title": "To Delete"})
        conv_id = json.loads(res_new.data)["conversation"]["id"]

        # Delete it
        res_del = self.client.delete(f"/api/conversations/{conv_id}")
        self.assertEqual(res_del.status_code, 200)

        # Confirm 404
        res_404 = self.client.get(f"/api/conversations/{conv_id}")
        self.assertEqual(res_404.status_code, 404)

        # Clear all
        res_clear = self.client.post("/api/conversations/clear")
        self.assertEqual(res_clear.status_code, 200)

        # Confirm empty list
        res_list = self.client.get("/api/conversations")
        data_list = json.loads(res_list.data)
        self.assertEqual(len(data_list["conversations"]), 0)


if __name__ == "__main__":
    unittest.main()
