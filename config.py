"""
config.py
Configuration and environment management for Mindora Mental Wellness Companion.
"""

import os
from datetime import timedelta

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Security
    SECRET_KEY = os.environ.get("MINDORA_SECRET_KEY", "mindora-serene-wellness-secret-key-2026")
    
    # Database
    DATABASE_PATH = os.environ.get(
        "MINDORA_DB_PATH", 
        os.path.join(BASE_DIR, "data", "mindora.db")
    )
    
    # Model and NLP Paths
    VOCAB_PATH = os.path.join(BASE_DIR, "data", "vocabulary.json")
    TAGS_PATH = os.path.join(BASE_DIR, "data", "tags.json")
    WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "lstm_weights.npz")
    INTENTS_PATH = os.path.join(BASE_DIR, "data", "intents.json")
    
    # ML & Conversation Parameters
    CONFIDENCE_THRESHOLD = float(os.environ.get("MINDORA_CONFIDENCE_THRESHOLD", 0.45))
    MAX_HISTORY_TURNS = int(os.environ.get("MINDORA_MAX_HISTORY_TURNS", 12))
    
    # Server settings
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 5000))
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(days=30)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
