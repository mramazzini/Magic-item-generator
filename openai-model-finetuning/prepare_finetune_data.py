import json
import jsonlines  


INPUT_PATH = "items_grouped_trimmed.json"
OUTPUT_PATH = "items_finetune.jsonl"


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    with jsonlines.open(OUTPUT_PATH, mode="w") as writer:
        for item in items:
            item_json_str = json.dumps(item, ensure_ascii=False)

            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a generator of structured Dungeons & Dragons items. "
                            "Always respond ONLY with a valid JSON object matching this exact schema:\n\n"
                            "{\n"
                            "  \"Name\": string,\n"
                            "  \"Description\": string,\n"
                            "  \"Features\": [\n"
                            "    {\n"
                            "      \"Name\": string or null,\n"
                            "      \"Description\": string\n"
                            "    }\n"
                            "  ]\n"
                            "}\n\n"
                            "Rules:\n"
                            "- Include ONLY the keys Name, Description, and Features.\n"
                            "- Features must be an array of objects with only Name and Description.\n"
                            "- No markdown. No extra commentary. No prefix text.\n"
                            "- Output must ALWAYS be a single JSON object with no wrapper text."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Generate a Dungeons & Dragons item in the required JSON format.",
                    },
                    {
                        "role": "assistant",
                        "content": item_json_str,
                    },
                ]
            }

            writer.write(example)

    print(f"Wrote fine-tune dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
