import psycopg2
from psycopg2.extras import RealDictCursor
import jsonlines

def get_connection():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="maxdnd",
        user="postgres",
        password="a",
        cursor_factory=RealDictCursor,
    )
    return conn


def fetch_all_items():
    query = 'SELECT * FROM "Items";'

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
        return rows
    finally:
        conn.close()


def fetch_all_item_features():
    query = 'SELECT * FROM "ItemFeatures";'

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
        return rows
    finally:
        conn.close()


def build_items_with_features():
    raw_items = fetch_all_items()
    items_by_id = {}

    for row in raw_items:
        item_id = row["Id"] 
        item_obj = dict(row) 
        item_obj["Features"] = []
        items_by_id[item_id] = item_obj

    raw_features = fetch_all_item_features()
    missing_parent_count = 0

    for feat in raw_features:
        item_id = feat.get("ItemId")
        if item_id in items_by_id:
            items_by_id[item_id]["Features"].append(dict(feat))
        else:
            missing_parent_count += 1

    if missing_parent_count > 0:
        print(f"Warning: {missing_parent_count} features had no matching item")

    return list(items_by_id.values())

def item_to_text(item: dict) -> str:
    parts = []

    name = item.get("Name") or ""
    desc = item.get("Description") or ""
    parts.append(f"Name: {name}")
    parts.append(f"Description: {desc}")

    features = item.get("Features") or []
    for feat in features:
        fname = feat.get("Name") or ""
        fdesc = feat.get("Description") or ""
        if fname:
            parts.append(f"Feature: {fname} — {fdesc}")
        else:
            parts.append(f"Feature: {fdesc}")

    return "\n".join(parts).strip()


def build_rarity_dataset(items, output_path="rarity_dataset.jsonl"):
    count = 0
    with jsonlines.open(output_path, mode="w") as writer:
        for item in items:
            rarity = item.get("Rarity")
            if rarity is None:
                continue

            text = item_to_text(item)
            if not text:
                continue

            writer.write({
                "text": text,
                "rarity": rarity
            })
            count += 1

    print(f"Wrote {count} examples to {output_path}")


def build_weight_dataset(items, output_path="weight_dataset.jsonl"):
    count = 0
    with jsonlines.open(output_path, mode="w") as writer:
        for item in items:
            weight = item.get("WeightPounds")
            if weight is None:
                continue

            text = item_to_text(item)
            if not text:
                continue

            writer.write({
                "text": text,
                "weight": weight
            })
            count += 1

    print(f"Wrote {count} examples to {output_path}")


def build_cost_dataset(items, output_path="cost_dataset.jsonl"):
    count = 0
    with jsonlines.open(output_path, mode="w") as writer:
        for item in items:
            cost = item.get("Cost")
            if cost is None:
                continue

            text = item_to_text(item)
            if not text:
                continue

            writer.write({
                "text": text,
                "cost": cost
            })
            count += 1

    print(f"Wrote {count} examples to {output_path}")


def build_tags_dataset(items, output_path="tags_dataset.jsonl"):
    count = 0
    with jsonlines.open(output_path, mode="w") as writer:
        for item in items:
            tags_raw = item.get("Tags")
            if not tags_raw:
                continue

            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            if not tags:
                continue

            text = item_to_text(item)
            if not text:
                continue

            writer.write({
                "text": text,
                "tags": tags
            })
            count += 1

    print(f"Wrote {count} examples to {output_path}")



def main():
    items = build_items_with_features()
    print(f"Loaded {len(items)} items from DB")

    build_rarity_dataset(items, "rarity_dataset.jsonl")
    build_weight_dataset(items, "weight_dataset.jsonl")
    build_cost_dataset(items, "cost_dataset.jsonl")
    build_tags_dataset(items, "tags_dataset.jsonl")


if __name__ == "__main__":
    main()
