from openai import OpenAI
from dotenv import load_dotenv
import os
import jsonlines
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import pickle

DATA_PATH = "tags_dataset.jsonl"
EMBEDDING_MODEL = "text-embedding-3-small"
MODEL_OUTPUT_PATH = "generated-classifiers/tags_classifier.pkl"

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_tags_dataset(path=DATA_PATH):
    texts = []
    tags_list = []

    with jsonlines.open(path, mode="r") as reader:
        for obj in reader:
            text = obj.get("text")
            tags = obj.get("tags")
            if not text or not tags:
                continue
            texts.append(text)
            tags_list.append([str(t) for t in tags])

    print(f"Loaded {len(texts)} examples from {path}")
    return texts, tags_list


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


def train_tags_classifier():
    texts, tags_list = load_tags_dataset(DATA_PATH)

    if not texts:
        print("No data found in tags dataset. Aborting.")
        return

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tags_list)
    print(f"Number of distinct tags: {len(mlb.classes_)}")

    X = embed_texts(texts)
    print(f"Embeddings shape: {X.shape}")
    print(f"Label matrix shape: {Y.shape}")

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, n_jobs=-1)
    )
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_val)

    micro_f1 = f1_score(Y_val, Y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(Y_val, Y_pred, average="macro", zero_division=0)
    print("\n=== TAGS MULTI-LABEL CLASSIFICATION REPORT ===")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nPer tag classification report :")
    print(classification_report(Y_val, Y_pred, target_names=mlb.classes_, zero_division=0))

    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(
            {
                "classifier": clf,
                "mlb": mlb,
                "embedding_model": EMBEDDING_MODEL,
            },
            f,
        )

    print(f"\nSaved tags classifier to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train_tags_classifier()
