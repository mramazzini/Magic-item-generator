from openai import OpenAI
from dotenv import load_dotenv
import os
import jsonlines
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import pickle

DATA_PATH = "rarity_dataset.jsonl"
EMBEDDING_MODEL = "text-embedding-3-small"
MODEL_OUTPUT_PATH = "../generated-classifiers/rarity_classifier.pkl"

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_rarity_dataset(path=DATA_PATH):
    texts = []
    labels = []

    with jsonlines.open(path, mode="r") as reader:
        for obj in reader:
            text = obj.get("text")
            rarity = obj.get("rarity")
            if not text or rarity is None:
                continue
            labels.append(str(rarity))
            texts.append(text)

    print(f"Loaded {len(texts)} examples from {path}")
    return texts, labels


def embed_texts(texts, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Embedding batch {i // batch_size + 1} / {int(np.ceil(len(texts) / batch_size))}...")
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embeddings = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings)


def train_rarity_classifier():
    texts, labels = load_rarity_dataset(DATA_PATH)

    if not texts:
        print("No data found in rarity dataset. Aborting.")
        return

    counts = Counter(labels)
    print("\nRarity distribution (before filtering):")
    for label, cnt in counts.items():
        print(f"  Rarity {label}: {cnt} examples")

    labels_filtered = []
    texts_filtered = []
    for text, label in zip(texts, labels):
        if counts[label] >= 2:
            texts_filtered.append(text)
            labels_filtered.append(label)

    if len(texts_filtered) < len(texts):
        print(f"\nFiltered out {len(texts) - len(texts_filtered)} examples from rare classes (<2 samples).")

    texts = texts_filtered
    labels = labels_filtered

    counts_filtered = Counter(labels)
    print("\nRarity distribution (after filtering):")
    for label, cnt in counts_filtered.items():
        print(f"  Rarity {label}: {cnt} examples")

    if len(set(labels)) < 2:
        print("\nNot enough classes to train a classifier (need at least 2). Aborting.")
        return

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X = embed_texts(texts)
    print(f"Embeddings shape: {X.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("\n=== RARITY CLASSIFICATION REPORT ===")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(
            {
                "classifier": clf,
                "label_encoder": label_encoder,
                "embedding_model": EMBEDDING_MODEL,
            },
            f,
        )

    print(f"\nSaved rarity classifier to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train_rarity_classifier()
