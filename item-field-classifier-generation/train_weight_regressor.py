from openai import OpenAI
from dotenv import load_dotenv
import os
import jsonlines
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

DATA_PATH = "weight_dataset.jsonl"
EMBEDDING_MODEL = "text-embedding-3-small"
MODEL_OUTPUT_PATH = "generated-classifiers/weight_regressor.pkl"

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_weight_dataset(path=DATA_PATH):
    texts = []
    weights = []

    with jsonlines.open(path, mode="r") as reader:
        for obj in reader:
            text = obj.get("text")
            weight = obj.get("weight")
            if text is None or weight is None:
                continue
            texts.append(text)
            weights.append(float(weight))

    print(f"Loaded {len(texts)} examples from {path}")
    return texts, weights


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


def train_weight_regressor():
    texts, weights = load_weight_dataset(DATA_PATH)

    if not texts:
        print("No data found in weight dataset. Aborting.")
        return

    X = embed_texts(texts)
    y = np.array(weights)
    print(f"Embeddings shape: {X.shape}")
    print(f"Targets shape: {y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print("\n=== WEIGHT REGRESSION REPORT ===")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R^2 Score:          {r2:.4f}")

    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(
            {
                "regressor": reg,
                "embedding_model": EMBEDDING_MODEL,
            },
            f,
        )

    print(f"\nSaved weight regressor to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train_weight_regressor()
