
from collections import Counter
from pathlib import Path
import hashlib
import json

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "3nesdeniz/agentic-prompt-injection-5k"
REPORT_DIR = Path(__file__).resolve().parents[1] / "data" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Download the dataset
dataset = load_dataset(DATASET_NAME)

print("\n========== DATASET SCHEMA ==========")

all_frames = {}

for split_name, split in dataset.items():
    df = split.to_pandas()
    all_frames[split_name] = df

    print(f"\nSplit: {split_name}")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes)

    # 2. Check class distribution
    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False))

    # 3. Missing and invalid data
    print("\nMissing values:")
    print(df.isnull().sum())

    empty_text = df["text"].fillna("").str.strip().eq("").sum()
    print(f"Empty text rows: {empty_text}")

    invalid_labels = (~df["label"].isin([0, 1])).sum()
    print(f"Invalid labels: {invalid_labels}")

    # 4. Detect exact duplicates
    duplicate_count = df["text"].duplicated().sum()
    print(f"Exact duplicate text rows: {duplicate_count}")

    # 5. Analyze attack families
    print("\nAttack families:")
    print(df["attack_family"].value_counts())

    # 6. Text length statistics
    lengths = df["text"].fillna("").str.len()
    print("\nText length statistics:")
    print(lengths.describe())

    # 7. Save class and family summaries
    df.groupby(["label", "attack_family"]).size().to_csv(
        REPORT_DIR / f"{split_name}_distribution.csv"
    )

# 8. Check exact duplicates across splits
print("\n========== CROSS-SPLIT LEAKAGE ==========")

hash_sets = {}

for name, df in all_frames.items():
    hashes = {
        hashlib.sha256(str(t).strip().encode("utf-8")).hexdigest()
        for t in df["text"].dropna()
    }
    hash_sets[name] = hashes

names = list(hash_sets)

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        overlap = hash_sets[a] & hash_sets[b]
        print(f"{a} vs {b}: {len(overlap)} exact overlaps")

print("\nInspection complete.")

# run command: python -m milestones.milestone1.scripts.inspect_dataset