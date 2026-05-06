"""ML module for grievance text classification."""

from __future__ import annotations

import os
import pickle
import re
from typing import List, Sequence, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODULE_DIR, "complaint_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODULE_DIR, "tfidf_vectorizer.pkl")

CATEGORIES = ["Water", "Electricity", "Roads", "Garbage", "Healthcare"]

# In-memory cache for loaded artifacts.
_MODEL: LogisticRegression | None = None
_VECTORIZER: TfidfVectorizer | None = None

_TRAINING_SAMPLES = {
    "Water": [
        "No water supply in my area since morning",
        "Pipeline leakage causing water wastage",
        "Drinking water is dirty and smells bad",
        "Water tanker has not arrived in two days",
        "Low water pressure in residential colony",
        "Public tap is broken near the market",
        "Water logging due to burst pipeline",
        "Water connection is not working at home",
    ],
    "Electricity": [
        "Power cut happening every night",
        "Electricity meter is not working properly",
        "Frequent voltage fluctuations damaged appliances",
        "Street lights are not working",
        "No electricity in our neighborhood",
        "Transformer failure caused blackout",
        "Short circuit sparks from electric pole",
        "High electricity bill issue needs verification",
    ],
    "Roads": [
        "Road is full of potholes and unsafe",
        "Main street is badly damaged after rain",
        "Broken road causing traffic congestion",
        "No proper drainage on roadside",
        "Bridge approach road has cracks",
        "Footpath is encroached and broken",
        "Road construction work stopped midway",
        "Street has uneven surface and accidents",
    ],
    "Garbage": [
        "Garbage not collected from our lane",
        "Overflowing dustbin near school",
        "Trash pile causing foul smell",
        "Waste collection vehicle not coming daily",
        "Open dumping attracting stray animals",
        "Unclean surroundings and plastic waste everywhere",
        "Sanitation workers are skipping our block",
        "Dead animal not removed from roadside",
    ],
    "Healthcare": [
        "Government hospital has no doctor available",
        "Primary health center lacks medicines",
        "Ambulance service is delayed in emergencies",
        "Clinic is overcrowded and unhygienic",
        "Vaccination camp not organized in village",
        "Need urgent medical help in community",
        "Hospital staff is not responding to patients",
        "Medical facility is too far and inaccessible",
    ],
}

_HEALTHCARE_KEYWORDS = {
    "ambulance",
    "clinic",
    "dawai",
    "dawa",
    "dispensary",
    "doctor",
    "emergency",
    "health",
    "healthcare",
    "hospital",
    "ilaj",
    "medical",
    "medicine",
    "nurse",
    "opd",
    "patient",
    "phc",
    "vaccine",
    "vaccination",
}

_GARBAGE_KEYWORDS = {
    "dustbin",
    "garbage",
    "kachra",
    "kooda",
    "litter",
    "safai",
    "sanitation",
    "trash",
    "waste",
}


def preprocess_text(text: str) -> str:
    """Normalize text: lowercase, remove punctuation, and remove stopwords."""
    if not text:
        return ""

    lowered = text.lower()
    letters_only = re.sub(r"[^a-z\s]", " ", lowered)
    tokens = [
        token
        for token in letters_only.split()
        if token and token not in ENGLISH_STOP_WORDS
    ]
    return " ".join(tokens)


def _build_training_dataset() -> Tuple[List[str], List[str]]:
    """Build feature and label lists for supervised training."""
    texts: List[str] = []
    labels: List[str] = []
    for category, samples in _TRAINING_SAMPLES.items():
        for sample in samples:
            texts.append(preprocess_text(sample))
            labels.append(category)
    return texts, labels


def _save_artifacts(model: LogisticRegression, vectorizer: TfidfVectorizer) -> None:
    """Persist trained artifacts to disk using pickle."""
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(VECTORIZER_PATH, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)


def _load_artifacts() -> Tuple[LogisticRegression, TfidfVectorizer]:
    """Load pickled model artifacts from disk."""
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


def train_model(force_retrain: bool = False) -> Tuple[LogisticRegression, TfidfVectorizer]:
    """Train model and save artifacts, or load existing artifacts."""
    global _MODEL, _VECTORIZER

    artifacts_exist = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
    if artifacts_exist and not force_retrain:
        _MODEL, _VECTORIZER = _load_artifacts()
        return _MODEL, _VECTORIZER

    train_texts, labels = _build_training_dataset()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    features = vectorizer.fit_transform(train_texts)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(features, labels)

    _save_artifacts(model=model, vectorizer=vectorizer)
    _MODEL, _VECTORIZER = model, vectorizer
    return model, vectorizer


def _keyword_based_category(text: str) -> str | None:
    """Apply rule-based fallback for high-confidence healthcare/garbage terms."""
    lowered = text.lower()
    healthcare_hits = sum(
        1 for keyword in _HEALTHCARE_KEYWORDS if keyword in lowered
    )
    garbage_hits = sum(
        1 for keyword in _GARBAGE_KEYWORDS if keyword in lowered
    )

    if healthcare_hits == 0 and garbage_hits == 0:
        return None
    if healthcare_hits >= garbage_hits:
        return "Healthcare"
    return "Garbage"


def classify_complaint(text: str) -> str:
    """Predict complaint category using trained ML artifacts."""
    global _MODEL, _VECTORIZER

    keyword_override = _keyword_based_category(text)
    if keyword_override is not None:
        return keyword_override

    if _MODEL is None or _VECTORIZER is None:
        train_model()

    processed = preprocess_text(text)
    if not processed:
        processed = "general issue"

    prediction = _MODEL.predict(_VECTORIZER.transform([processed]))[0]
    return str(prediction)
