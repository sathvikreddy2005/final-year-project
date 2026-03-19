import os
import json
import numpy as np
import time

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import lightgbm as lgb

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel


# --------------------------------------------------
# PATHS
# --------------------------------------------------
DATA_FOLDER = "data/final-eriskt2-dataset-with-ground-truth/all_combined"
LABEL_FILE = "data/final-eriskt2-dataset-with-ground-truth/shuffled_ground_truth_labels.txt"

EMBED_FILE = "saved_embeddings.npy"
LABELS_FILE = "saved_labels.npy"


# --------------------------------------------------
# LOAD LABELS
# --------------------------------------------------
def load_labels(label_path):
    labels = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                subject_id, label = parts
                labels[subject_id] = int(label)
    return labels


# --------------------------------------------------
# COMBINE USER POSTS
# --------------------------------------------------
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


# --------------------------------------------------
# BUILD DATASET
# --------------------------------------------------
def build_dataset():
    labels = load_labels(LABEL_FILE)

    X = []
    y = []

    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".json"):
            subject_id = filename.replace(".json", "")
            if subject_id in labels:
                file_path = os.path.join(DATA_FOLDER, filename)
                user_text = combine_user_posts(file_path)

                if len(user_text) > 0:
                    X.append(user_text)
                    y.append(labels[subject_id])

    return X, np.array(y)


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
if __name__ == "__main__":

    start_time = time.time()

    print("Loading dataset...")
    X, y = build_dataset()
    print("Total users:", len(X))
    print("Class distribution:", np.bincount(y))

    # --------------------------------------------------
    # SBERT (Compute once and save)
    # --------------------------------------------------
    if os.path.exists(EMBED_FILE):
        print("\nLoading saved embeddings...")
        embeddings = np.load(EMBED_FILE)
        y = np.load(LABELS_FILE)
    else:
        print("\nGenerating SBERT embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(X, show_progress_bar=True)

        np.save(EMBED_FILE, embeddings)
        np.save(LABELS_FILE, y)

    print("Embedding shape:", embeddings.shape)

    # --------------------------------------------------
    # PCA (Reduce for Quantum Feasibility)
    # --------------------------------------------------
    pca = PCA(n_components=4)
    X_reduced = pca.fit_transform(embeddings)

    # --------------------------------------------------
    # Scaling
    # --------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    # --------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ==================================================
    # 1️⃣ LIGHTGBM (PRIMARY MODEL)
    # ==================================================
    print("\nTraining LightGBM...")

    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )

    lgb_model.fit(X_train, y_train)

    lgb_pred = lgb_model.predict(X_test)
    lgb_prob = lgb_model.predict_proba(X_test)[:, 1]

    print("\nLightGBM Accuracy:", accuracy_score(y_test, lgb_pred))
    print("\nLightGBM Report:")
    print(classification_report(y_test, lgb_pred))


    # ==================================================
    # 2️⃣ QUANTUM QSVC (EXPERIMENTAL MODEL)
    #    Train on subset for speed
    # ==================================================
    print("\nTraining Quantum QSVC (subset mode)...")

    subset_size = 200
    X_train_small = X_train[:subset_size]
    y_train_small = y_train[:subset_size]

    feature_map = ZZFeatureMap(feature_dimension=4)

    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    qsvc = QSVC(quantum_kernel=quantum_kernel)

    qsvc.fit(X_train_small, y_train_small)

    q_pred = qsvc.predict(X_test)

    print("\nQuantum Accuracy:", accuracy_score(y_test, q_pred))
    print("\nQuantum Report:")
    print(classification_report(y_test, q_pred))


    # ==================================================
    # TEXT MODULE OUTPUT (FOR FUSION)
    # ==================================================
    print("\nGenerating Final Text Probability Score (from LightGBM)...")

    text_scores = lgb_prob   # Use LightGBM probabilities as final text score

    print("Sample Text Scores:", text_scores[:5])


    end_time = time.time()
    print("\nTotal Execution Time (seconds):", round(end_time - start_time, 2))