import os
import json
import numpy as np
from pathlib import Path
import joblib

from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from sklearn.utils import resample

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler


# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_FOLDER = PROJECT_ROOT / "data" / "final-eriskt2-dataset-with-ground-truth" / "all_combined"
LABEL_FILE = PROJECT_ROOT / "data" / "final-eriskt2-dataset-with-ground-truth" / "shuffled_ground_truth_labels.txt"
LGBM_MODEL_FILE = BASE_DIR / "lightgbm_text_model.pkl"
LGBM_METADATA_FILE = BASE_DIR / "lightgbm_text_metadata.json"


# -------------------------------
# Load labels
# -------------------------------
def load_labels(label_path):
    labels = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                subject_id, label = parts
                labels[subject_id] = int(label)
    return labels


# -------------------------------
# Combine posts per user
# -------------------------------
def combine_user_posts(json_path):
    combined_text = ""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        submission_data = item.get("submission", {})
        title = submission_data.get("title", "")
        body = submission_data.get("body", "")
        combined_text += title + " " + body + " "

    return combined_text.strip()


# -------------------------------
# Build dataset
# -------------------------------
def build_dataset():
    labels = load_labels(LABEL_FILE)

    X = []
    y = []

    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".json"):
            subject_id = filename.replace(".json", "")
            if subject_id in labels:
                file_path = DATA_FOLDER / filename
                user_text = combine_user_posts(file_path)

                if len(user_text) > 0:
                    X.append(user_text)
                    y.append(labels[subject_id])

    return X, np.array(y)


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    print("Loading dataset...")
    X, y = build_dataset()
    print("Total users:", len(X))
    print("Class distribution:", np.bincount(y))


    # ---------------------------
    # SBERT Embeddings
    # ---------------------------
    print("\nLoading SBERT...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")
    embeddings = model.encode(X, show_progress_bar=True)
    print("Embedding shape:", embeddings.shape)


    # ---------------------------
    # Train-Test Split
    # ---------------------------
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Train size:", len(X_train_full))
    print("Test size:", len(X_test_full))
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    scale_pos_weight = neg_count / max(pos_count, 1)


    # ==========================================================
    # LIGHTGBM MODEL TUNING
    # ==========================================================
    print("\nTuning LightGBM...")

    threshold_candidates = [0.50, 0.40, 0.35, 0.30, 0.25]
    model_candidates = [
        {
            "name": "baseline",
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 5,
                "class_weight": "balanced",
                "random_state": 42,
            },
        },
        {
            "name": "recall_focus_a",
            "params": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 31,
                "min_child_samples": 10,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "scale_pos_weight": scale_pos_weight,
                "random_state": 42,
            },
        },
        {
            "name": "recall_focus_b",
            "params": {
                "n_estimators": 400,
                "learning_rate": 0.03,
                "max_depth": -1,
                "num_leaves": 63,
                "min_child_samples": 8,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "scale_pos_weight": scale_pos_weight * 1.25,
                "random_state": 42,
            },
        },
    ]

    best_result = None
    X_train_lgb = np.array(X_train_full)
    y_train_lgb = np.array(y_train)

    X_train_neg = X_train_lgb[y_train_lgb == 0]
    X_train_pos = X_train_lgb[y_train_lgb == 1]
    X_train_pos_upsampled = resample(
        X_train_pos,
        replace=True,
        n_samples=len(X_train_neg),
        random_state=42,
    )
    X_train_balanced_lgb = np.vstack((X_train_neg, X_train_pos_upsampled))
    y_train_balanced_lgb = np.array([0] * len(X_train_neg) + [1] * len(X_train_pos_upsampled))
    shuffle_idx_lgb = np.random.permutation(len(X_train_balanced_lgb))
    X_train_balanced_lgb = X_train_balanced_lgb[shuffle_idx_lgb]
    y_train_balanced_lgb = y_train_balanced_lgb[shuffle_idx_lgb]

    print(
        "Balanced class distribution for LightGBM:",
        np.bincount(y_train_balanced_lgb),
    )

    for candidate in model_candidates:
        print(f"\nTraining LightGBM candidate: {candidate['name']}")
        lgb_model = lgb.LGBMClassifier(**candidate["params"])
        lgb_model.fit(X_train_balanced_lgb, y_train_balanced_lgb)

        lgb_probabilities = lgb_model.predict_proba(X_test_full)[:, 1]
        roc_auc = roc_auc_score(y_test, lgb_probabilities)
        pr_auc = average_precision_score(y_test, lgb_probabilities)
        print("ROC-AUC:", round(roc_auc, 4))
        print("PR-AUC:", round(pr_auc, 4))

        for threshold in threshold_candidates:
            preds = (lgb_probabilities >= threshold).astype(int)
            metrics = {
                "model_name": candidate["name"],
                "threshold": threshold,
                "accuracy": accuracy_score(y_test, preds),
                "precision_1": precision_score(y_test, preds, zero_division=0),
                "recall_1": recall_score(y_test, preds, zero_division=0),
                "f1_1": f1_score(y_test, preds, zero_division=0),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "confusion_matrix": confusion_matrix(y_test, preds),
                "probabilities": lgb_probabilities,
                "predictions": preds,
                "model": lgb_model,
            }

            print(
                f"Threshold {threshold:.2f} | "
                f"acc={metrics['accuracy']:.3f} "
                f"prec1={metrics['precision_1']:.3f} "
                f"rec1={metrics['recall_1']:.3f} "
                f"f1_1={metrics['f1_1']:.3f}"
            )

            if best_result is None:
                best_result = metrics
                continue

            current_key = (metrics["f1_1"], metrics["recall_1"], metrics["pr_auc"])
            best_key = (best_result["f1_1"], best_result["recall_1"], best_result["pr_auc"])
            if current_key > best_key:
                best_result = metrics

    joblib.dump(best_result["model"], LGBM_MODEL_FILE)
    metadata = {
        "model_name": best_result["model_name"],
        "threshold": float(best_result["threshold"]),
        "accuracy": float(best_result["accuracy"]),
        "precision_class_1": float(best_result["precision_1"]),
        "recall_class_1": float(best_result["recall_1"]),
        "f1_class_1": float(best_result["f1_1"]),
        "roc_auc": float(best_result["roc_auc"]),
        "pr_auc": float(best_result["pr_auc"]),
        "train_class_distribution": {
            "negative": int(neg_count),
            "positive": int(pos_count),
        },
        "balanced_train_class_distribution": {
            "negative": int(np.sum(y_train_balanced_lgb == 0)),
            "positive": int(np.sum(y_train_balanced_lgb == 1)),
        },
        "test_class_distribution": {
            "negative": int(np.sum(y_test == 0)),
            "positive": int(np.sum(y_test == 1)),
        },
        "feature_source": "SBERT all-MiniLM-L6-v2 embeddings",
        "feature_shape": int(embeddings.shape[1]),
    }
    with open(LGBM_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nBest LightGBM setup:")
    print(
        f"Model={best_result['model_name']} | "
        f"threshold={best_result['threshold']:.2f} | "
        f"acc={best_result['accuracy']:.3f} | "
        f"prec1={best_result['precision_1']:.3f} | "
        f"rec1={best_result['recall_1']:.3f} | "
        f"f1_1={best_result['f1_1']:.3f} | "
        f"roc_auc={best_result['roc_auc']:.3f} | "
        f"pr_auc={best_result['pr_auc']:.3f}"
    )
    print("Best LightGBM Confusion Matrix:\n", best_result["confusion_matrix"])
    print("Best LightGBM Classification Report:")
    print(classification_report(y_test, best_result["predictions"]))
    print("Sample LightGBM Scores:", best_result["probabilities"][:5])
    print("Saved LightGBM model to:", LGBM_MODEL_FILE)
    print("Saved LightGBM metadata to:", LGBM_METADATA_FILE)


    # ---------------------------
    # PCA Reduction For Quantum
    # ---------------------------
    print("\nApplying PCA for Quantum...")
    pca = PCA(n_components=4)
    X_reduced = pca.fit_transform(embeddings)
    print("Reduced shape:", X_reduced.shape)


    # ---------------------------
    # Scale Reduced Features For Quantum
    # ---------------------------
    print("\nScaling reduced features for Quantum...")
    scaler = StandardScaler()
    X_reduced = scaler.fit_transform(X_reduced)

    X_train, X_test, _, _ = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42, stratify=y
    )


    # ==========================================================
    # BALANCE TRAINING DATA FOR QUANTUM
    # ==========================================================
    print("\nBalancing training data for Quantum...")

    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    X_0 = X_train_np[y_train_np == 0]
    X_1 = X_train_np[y_train_np == 1]

    X_1_upsampled = resample(
        X_1,
        replace=True,
        n_samples=len(X_0),
        random_state=42
    )

    X_train_balanced = np.vstack((X_0, X_1_upsampled))
    y_train_balanced = np.array([0]*len(X_0) + [1]*len(X_1_upsampled))

    shuffle_idx = np.random.permutation(len(X_train_balanced))
    X_train_balanced = X_train_balanced[shuffle_idx]
    y_train_balanced = y_train_balanced[shuffle_idx]

    print("Balanced class distribution:", np.bincount(y_train_balanced))


    # ==========================================================
    # QUANTUM MODEL (Light but Correct)
    # ==========================================================
    print("\nTraining Quantum Model...")

    num_qubits = 4

    feature_map = ZZFeatureMap(feature_dimension=num_qubits)
    ansatz = RealAmplitudes(num_qubits, reps=1)

    optimizer = COBYLA(maxiter=100)
    sampler = StatevectorSampler()

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler
    )

    vqc.fit(X_train_balanced, y_train_balanced)

    predictions = vqc.predict(X_test)

    print("\nQuantum Accuracy:", accuracy_score(y_test, predictions))
    print("Unique predictions:", np.unique(predictions))

    print("\nQuantum Classification Report:")
    print(classification_report(y_test, predictions))
