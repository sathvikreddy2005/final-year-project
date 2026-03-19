import os
import json

# 🔹 Change these paths if needed
DATA_FOLDER = r"data\final-eriskt2-dataset-with-ground-truth\all_combined"
LABEL_FILE = r"data\final-eriskt2-dataset-with-ground-truth\shuffled_ground_truth_labels.txt"


# ------------------------------------------------
# STEP 1: Load ground truth labels
# ------------------------------------------------
def load_labels(label_path):
    labels = {}

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                subject_id, label = parts
                labels[subject_id] = int(label)

    return labels


# ------------------------------------------------
# STEP 2: Combine all posts for ONE user
# ------------------------------------------------
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



# ------------------------------------------------
# STEP 3: Build full dataset
# ------------------------------------------------
def build_dataset():
    labels = load_labels(LABEL_FILE)

    X = []  # Text data
    y = []  # Labels

    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".json"):
            subject_id = filename.replace(".json", "")

            if subject_id in labels:
                file_path = os.path.join(DATA_FOLDER, filename)

                user_text = combine_user_posts(file_path)

                X.append(user_text)
                y.append(labels[subject_id])

    return X, y


# ------------------------------------------------
# MAIN
# ------------------------------------------------
if __name__ == "__main__":
    X, y = build_dataset()

    print("Total users loaded:", len(X))
    print("Example text length:", len(X[0]))
    print("Example label:", y[0])
