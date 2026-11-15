
from collections import Counter
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

dataset = load_dataset(
    "3nesdeniz/agentic-prompt-injection-5k"
)

train = dataset["train"]
validation = dataset["validation"]

# Step 1: Learn the majority class from TRAIN only.
counts = Counter(train["label"])
majority_label = counts.most_common(1)[0][0]

print("Majority class:", majority_label)

# Step 2: Predict the same class for every sample.
y_true = validation["label"]
y_pred = [majority_label] * len(validation)

# Step 3: Evaluate.
tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if fp + tn else 0.0

print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall:", recall_score(y_true, y_pred, zero_division=0))
print("F1:", f1_score(y_true, y_pred, zero_division=0))
print("FPR:", fpr)

# run command: python -m milestones.milestone1.scripts.majority_baseline
