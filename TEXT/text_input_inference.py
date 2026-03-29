import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "lightgbm_text_model.pkl"
METADATA_FILE = ROOT / "lightgbm_text_metadata.json"

_EMBEDDER = None
_MODEL = None


def load_runtime_objects():
    global _EMBEDDER, _MODEL

    if _MODEL is None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Text model file not found: {MODEL_FILE}")
        _MODEL = joblib.load(MODEL_FILE)

    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

    return _EMBEDDER, _MODEL


def level_from_score(score: float) -> str:
    if score < 35:
        return "low"
    if score < 65:
        return "moderate"
    return "high"


def predict_levels(text: str) -> Dict[str, Dict[str, object]]:
    embedder, model = load_runtime_objects()
    embedding = embedder.encode([text], show_progress_bar=False)
    depression_prob = float(model.predict_proba(embedding)[0, 1]) * 100.0

    return {
        "depression": {
            "score": round(depression_prob, 2),
            "level": level_from_score(depression_prob),
            "source": "text_model",
        },
    }


def load_metadata() -> Dict[str, object]:
    if not METADATA_FILE.exists():
        return {}
    return json.loads(METADATA_FILE.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Predict stress levels from text input."
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="User input text for inference.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only JSON output (for API integration).",
    )
    args = parser.parse_args()

    user_text = args.text.strip()

    result = predict_levels(user_text)
    metadata = load_metadata()
    if metadata.get("threshold") is not None:
        result["metadata"] = {
            "threshold": metadata["threshold"],
            "model_name": metadata.get("model_name", "lightgbm"),
        }

    if args.json_only:
        print(json.dumps(result))
    else:
        print("\nText Mental Health Output")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
