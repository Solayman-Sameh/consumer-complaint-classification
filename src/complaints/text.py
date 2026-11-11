"""Versioned text normalization shared by preparation and inference."""
import hashlib
import re
import unicodedata

PREPROCESSING_VERSION = "complaints-v1"
PAD_ID = 0
UNK_ID = 1


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def clean_recurrent(text, stopwords):
    text = re.sub(r"[^a-z]+", " ", normalize_text(text))
    return " ".join(word for word in text.split() if word not in stopwords)


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encode_recurrent(text, vocabulary, stopwords, max_length):
    words = clean_recurrent(text, stopwords).split()
    if not words:
        raise ValueError("No usable words remain after preprocessing.")
    return [vocabulary.get(word, UNK_ID) for word in words[:max_length]]
