import sys
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

from predictors.rarity_classifier import predict_rarity_for_item
from predictors.weight_predictor import predict_weight_for_item
from predictors.cost_predictor import predict_cost_for_item
from predictors.tags_predictor import predict_tags_for_item

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:maxdnd::ClJwDG03"

def generate_item(prompt: str):
    response = client.chat.completions.create(
        model=FINE_TUNED_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a generator of structured Dungeons & Dragons items. "
                    "Always respond ONLY with JSON containing Name, Description, "
                    "and Features."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def enrich_item(item):
    item["PredictedRarity"] = predict_rarity_for_item(item)
    item["PredictedWeight"] = predict_weight_for_item(item)
    item["PredictedCost"] = predict_cost_for_item(item)
    item["PredictedTags"] = predict_tags_for_item(item)
    return item


def save_item_to_file(item, filename="generated_item.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(item, f, indent=4, ensure_ascii=False)
    print(f"\nSaved item to {filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate.py \"Your item prompt here\"")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])

    print(f"\nGenerating item from prompt:\n→ {prompt}\n")

    item = generate_item(prompt)
    enriched_item = enrich_item(item)

    print("\n=== GENERATED & CLASSIFIED ITEM ===\n")
    print(json.dumps(enriched_item, indent=4, ensure_ascii=False))

    save_item_to_file(enriched_item, "output/generated_item.json")
