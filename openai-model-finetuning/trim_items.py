import psycopg2
from psycopg2.extras import RealDictCursor
import json


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


def test_connection():
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("SELECT version();")
        row = cur.fetchone()

        print("Connected to Postgres!")
        print("Postgres version:", row["version"])

        cur.close()
        conn.close()
        print("Connection closed cleanly.")

    except Exception as e:
        print("Error connecting to the database:")
        print(e)


def fetch_all_items():
    query = 'SELECT * FROM "Items";'

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()  # list of RealDictRow
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

    print(f"Loaded {len(items_by_id)} items")

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


def trim_fields(items):
    item_fields_to_remove = {
        "Id",
        "Rarity",
        "Slug",
        "WeightPounds",
        "Cost",
        "Focus",
        "ToolId",
        "WeaponId",
        "ArmorId",
        "Visibility",
        "Charges",
        "Tags",
        "Icon"
    }

    feature_fields_to_keep = {"Name", "Description"}

    for item in items:
        for key in list(item.keys()):
            if key in item_fields_to_remove:
                item.pop(key, None)

        cleaned_features = []
        for feat in item.get("Features", []):
            cleaned_feat = {k: feat.get(k) for k in feature_fields_to_keep}
            cleaned_features.append(cleaned_feat)

        item["Features"] = cleaned_features

    return items


def dump_items_with_features_to_json(json_path="items_grouped_trimmed.json"):
    items = build_items_with_features()
    trimmed_items = trim_fields(items)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(trimmed_items, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(trimmed_items)} cleaned items to {json_path}")


if __name__ == "__main__":
    test_connection()
    dump_items_with_features_to_json("items_grouped_trimmed.json")
