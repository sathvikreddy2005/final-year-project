import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_FOLDER = PROJECT_ROOT / "data" / "final-eriskt2-dataset-with-ground-truth" / "all_combined"
LABEL_FILE = PROJECT_ROOT / "data" / "final-eriskt2-dataset-with-ground-truth" / "shuffled_ground_truth_labels.txt"
EMBED_FILE = PROJECT_ROOT / "saved_embeddings.npy"
LABELS_FILE = PROJECT_ROOT / "saved_labels.npy"


def load_labels(label_path: str) -> Dict[str, int]:
    labels = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                subject_id, label = parts
                labels[subject_id] = int(label)
    return labels


def combine_user_posts(json_path: str) -> str:
    combined_text = ""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        submission_data = item.get("submission", {})
        title = submission_data.get("title", "")
        body = submission_data.get("body", "")
        combined_text += title + " " + body + " "

    return combined_text.strip()


def build_dataset() -> Tuple[List[str], np.ndarray]:
    labels = load_labels(LABEL_FILE)
    X, y = [], []

    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".json"):
            subject_id = filename.replace(".json", "")
            if subject_id in labels:
                file_path = DATA_FOLDER / filename
                user_text = combine_user_posts(file_path)
                if user_text:
                    X.append(user_text)
                    y.append(labels[subject_id])

    return X, np.array(y)


def fit_text_risk_pipeline():
    if os.path.exists(EMBED_FILE) and os.path.exists(LABELS_FILE):
        embeddings = np.load(EMBED_FILE)
        y = np.load(LABELS_FILE)
    else:
        X, y = build_dataset()
        sbert_for_dataset = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sbert_for_dataset.encode(X, show_progress_bar=True)
        np.save(EMBED_FILE, embeddings)
        np.save(LABELS_FILE, y)

    pca = PCA(n_components=4, random_state=42)
    X_reduced = pca.fit_transform(embeddings)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_scaled, y)

    # Separate encoder for runtime text input
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return embedder, pca, scaler, clf


def level_from_score(score: float) -> str:
    if score < 35:
        return "low"
    if score < 65:
        return "moderate"
    return "high"


def proxy_score(text: str, keywords: List[str], fallback: float) -> float:
    t = text.lower()
    hits = sum(t.count(word) for word in keywords)
    # Saturating keyword boost (0 to ~30) + fallback influence.
    score = fallback * 0.6 + min(30.0, 7.5 * hits)
    return float(max(0.0, min(100.0, score)))


def predict_levels(text: str, embedder, pca, scaler, clf) -> Dict[str, Dict[str, object]]:
    emb = embedder.encode([text], show_progress_bar=False)
    features = scaler.transform(pca.transform(emb))
    depression_prob = float(clf.predict_proba(features)[0, 1]) * 100.0

    stress_keywords = [
        "stressed",
        "pressure",
        "overwhelmed",
        "burnout",
        "deadline",
        "tired",
        "restless",
        "irritated",
    ]
    anxiety_keywords = [
        "anxious",
        "worry",
        "worried",
        "panic",
        "fear",
        "nervous",
        "uneasy",
        "racing thoughts",
    ]

    stress_score = proxy_score(text, stress_keywords, fallback=depression_prob * 0.65)
    anxiety_score = proxy_score(text, anxiety_keywords, fallback=depression_prob * 0.75)

    return {
        "stress": {
            "score": round(stress_score, 2),
            "level": level_from_score(stress_score),
            "source": "proxy_from_text_keywords",
        },
        "depression": {
            "score": round(depression_prob, 2),
            "level": level_from_score(depression_prob),
            "source": "text_model",
        },
        "anxiety": {
            "score": round(anxiety_score, 2),
            "level": level_from_score(anxiety_score),
            "source": "proxy_from_text_keywords",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict stress, depression, and anxiety levels from text input."
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="User input text. If omitted, interactive input is used.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only JSON output (for API integration).",
    )
    args = parser.parse_args()

    user_text = args.text.strip()
    if not user_text:
        user_text = input("Enter user text: ").strip()

    if len(user_text) < 10:
        raise ValueError("Input text is too short. Please provide at least 10 characters.")

    embedder, pca, scaler, clf = fit_text_risk_pipeline()
    result = predict_levels(user_text, embedder, pca, scaler, clf)

    if args.json_only:
        print(json.dumps(result))
    else:
        print("\nText Mental Health Output")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
