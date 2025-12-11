from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TRAINING_FILE_PATH = "items_finetune.jsonl"


def main():
    with open(TRAINING_FILE_PATH, "rb") as f:
        uploaded = client.files.create(
            file=f,
            purpose="fine-tune",
        )

    print("Uploaded file:")
    print("  id:   ", uploaded.id)
    print("  name: ", uploaded.filename)


if __name__ == "__main__":
    main()
