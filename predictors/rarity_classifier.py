from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
import pickle

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CLASSIFIER_PATH = "generated-classifiers/rarity_classifier.pkl"

with open(CLASSIFIER_PATH, "rb") as f:
    saved = pickle.load(f)

RARITY_CLF = saved["classifier"]
RARITY_LABEL_ENCODER = saved["label_encoder"]
EMBEDDING_MODEL = saved["embedding_model"] 


def item_to_text(item: dict) -> str:
    parts = []

    name = item.get("Name") or ""
    desc = item.get("Description") or ""
    parts.append(f"Name: {name}")
    parts.append(f"Description: {desc}")

    features = item.get("Features") or []
    for feat in features:
        if isinstance(feat, dict):
            fname = feat.get("Name") or ""
            fdesc = feat.get("Description") or ""
            if fname:
                parts.append(f"Feature: {fname} — {fdesc}")
            else:
                parts.append(f"Feature: {fdesc}")
        else:
            parts.append(f"Feature: {str(feat)}")

    return "\n".join(parts).strip()


def embed_text(text: str):
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def predict_rarity_for_item(item: dict) -> str:
    text = item_to_text(item)
    embedding = embed_text(text)

    X = np.array(embedding).reshape(1, -1)
    y_pred = RARITY_CLF.predict(X)[0]
    label = RARITY_LABEL_ENCODER.inverse_transform([y_pred])[0]
    return label
