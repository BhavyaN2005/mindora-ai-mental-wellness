"""
preprocess.py
Text preprocessing and vocabulary management for Mental Health Chatbot.
STRICT REQUIREMENT: Uses ONLY Python standard library (json, re, os).
"""

import json
import os
import re
from typing import Dict, List, Tuple, Any, Optional

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1
DEFAULT_MAX_SEQ_LEN = 20


def clean_text(text: str) -> str:
    """
    Cleans raw text:
    - Converts to lowercase
    - Normalizes common contractions
    - Strips unwanted characters while preserving alphanumeric tokens
    - Collapses consecutive whitespace
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    
    # Standardize common contractions
    contractions = {
        r"can't": "cannot",
        r"won't": "will not",
        r"n't": " not",
        r"'re": " are",
        r"'s": " is",
        r"'d": " would",
        r"'ll": " will",
        r"'ve": " have",
        r"'m": " am",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
    
    # Remove non-alphanumeric characters (keep basic letters, digits and whitespace)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text into a list of words using Python's built-in re module.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()


class Preprocessor:
    def __init__(self, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.word2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word: Dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self.tag2idx: Dict[str, int] = {}
        self.idx2tag: Dict[int, str] = {}
        self.responses_by_tag: Dict[str, List[str]] = {}
        self.vocab_size: int = 2
        self.num_classes: int = 0

    def fit_from_json(self, json_path: str) -> Tuple[int, int, int]:
        """
        Parses intents.json, extracts non-empty patterns and tags,
        builds vocabulary and label encodings.
        Returns (num_samples, vocab_size, num_classes).
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Intents file not found at: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        intents = data.get("intents", [])
        if not intents:
            raise ValueError("No 'intents' list found in dataset.")

        words_seen = set()
        valid_tags = []
        sample_count = 0

        for intent in intents:
            tag = intent.get("tag")
            if not tag or not isinstance(tag, str):
                continue

            patterns = intent.get("patterns", [])
            responses = intent.get("responses", [])

            # Filter out empty or whitespace-only patterns
            clean_patterns = [p for p in patterns if isinstance(p, str) and clean_text(p)]
            
            # Filter valid responses
            clean_responses = [r for r in responses if isinstance(r, str) and r.strip()]
            if not clean_responses:
                clean_responses = ["I hear you. Could you please tell me more about how you're feeling?"]

            self.responses_by_tag[tag] = clean_responses

            if clean_patterns:
                valid_tags.append(tag)
                for pat in clean_patterns:
                    sample_count += 1
                    tokens = tokenize(pat)
                    words_seen.update(tokens)

        # Sort words and tags for deterministic ordering
        sorted_words = sorted(list(words_seen))
        for word in sorted_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        self.vocab_size = len(self.word2idx)

        # Build tag encodings
        unique_tags = sorted(list(set(valid_tags)))
        self.tag2idx = {tag: i for i, tag in enumerate(unique_tags)}
        self.idx2tag = {i: tag for i, tag in enumerate(unique_tags)}
        self.num_classes = len(self.tag2idx)

        return sample_count, self.vocab_size, self.num_classes

    def text_to_sequence(self, text: str) -> Tuple[List[int], int]:
        """
        Converts a text string into a padded integer sequence and returns
        (padded_sequence, actual_length).
        """
        tokens = tokenize(text)
        seq = [self.word2idx.get(w, UNK_IDX) for w in tokens]
        actual_len = min(len(seq), self.max_seq_len)
        if actual_len == 0:
            # Handle empty tokenization
            seq = [UNK_IDX]
            actual_len = 1

        # Truncate or pad to max_seq_len
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]
        else:
            seq = seq + [PAD_IDX] * (self.max_seq_len - len(seq))

        return seq, actual_len

    def prepare_dataset(self, json_path: str):
        """
        Prepares training arrays (X, y, lengths) from intents.json.
        Returns:
            X: List[List[int]] of shape (N, max_seq_len)
            y: List[int] of shape (N,)
            lengths: List[int] of shape (N,)
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        X, y, lengths = [], [], []

        for intent in data.get("intents", []):
            tag = intent.get("tag")
            if tag not in self.tag2idx:
                continue
            tag_idx = self.tag2idx[tag]

            for pat in intent.get("patterns", []):
                if not isinstance(pat, str) or not clean_text(pat):
                    continue
                seq, actual_len = self.text_to_sequence(pat)
                X.append(seq)
                y.append(tag_idx)
                lengths.append(actual_len)

        return X, y, lengths

    def save_metadata(self, vocab_path: str, tags_path: str):
        """
        Saves vocabulary and tags mapping to JSON files.
        """
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        os.makedirs(os.path.dirname(tags_path), exist_ok=True)

        vocab_data = {
            "word2idx": self.word2idx,
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)

        tags_data = {
            "tag2idx": self.tag2idx,
            "idx2tag": {str(k): v for k, v in self.idx2tag.items()},
            "responses_by_tag": self.responses_by_tag,
            "num_classes": self.num_classes
        }
        with open(tags_path, "w", encoding="utf-8") as f:
            json.dump(tags_data, f, indent=2)

    def load_metadata(self, vocab_path: str, tags_path: str):
        """
        Loads saved vocabulary and tags mapping from JSON files.
        """
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        self.word2idx = vocab_data["word2idx"]
        self.idx2word = {int(idx): word for word, idx in self.word2idx.items()}
        self.max_seq_len = vocab_data.get("max_seq_len", DEFAULT_MAX_SEQ_LEN)
        self.vocab_size = vocab_data.get("vocab_size", len(self.word2idx))

        with open(tags_path, "r", encoding="utf-8") as f:
            tags_data = json.load(f)
        self.tag2idx = tags_data["tag2idx"]
        self.idx2tag = {int(k): v for k, v in tags_data["idx2tag"].items()}
        self.responses_by_tag = tags_data.get("responses_by_tag", {})
        self.num_classes = tags_data.get("num_classes", len(self.tag2idx))
