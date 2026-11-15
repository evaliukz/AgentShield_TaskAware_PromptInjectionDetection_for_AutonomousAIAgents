
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

dataset = load_dataset(
    "3nesdeniz/agentic-prompt-injection-5k"
)

train_texts = dataset["train"]["text"]

# Character n-grams capture similar wording,
# even when a few characters or words change.
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=1,
    sublinear_tf=True
)

train_vectors = vectorizer.fit_transform(train_texts)

neighbors = NearestNeighbors(
    n_neighbors=1,
    metric="cosine",
    algorithm="brute"
)
neighbors.fit(train_vectors)

for split_name in ["validation", "test"]:
    texts = dataset[split_name]["text"]
    vectors = vectorizer.transform(texts)

    distances, indices = neighbors.kneighbors(vectors)
    similarities = 1 - distances[:, 0]

    print(f"\n{split_name.upper()} vs TRAIN")

    for threshold in [0.80, 0.90, 0.95]:
        count = np.sum(similarities >= threshold)
        print(f"Similarity >= {threshold}: {count}")

    print(f"Maximum similarity: {similarities.max():.3f}")
    print(f"Median similarity: {np.median(similarities):.3f}")

    # Show the top 3 most similar pairs.
    top_indices = np.argsort(similarities)[-3:][::-1]

    for i in top_indices:
        print(f"\nSimilarity: {similarities[i]:.3f}")
        print("Candidate:", texts[int(i)])
        print("Train:", train_texts[int(indices[i, 0])])

# python -m pip install numpy scipy
# run command: python -m milestones.milestone1.scripts.check_leakage