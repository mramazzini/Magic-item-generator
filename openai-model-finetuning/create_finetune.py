from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TRAINING_FILE_ID = "file-QMKJsnVhD4DdFF49rkHVWM"  


def main():
    job = client.fine_tuning.jobs.create(
        model="gpt-4o-mini-2024-07-18",   
        training_file=TRAINING_FILE_ID,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": 3,
                }
            },
        },
        metadata={"project": "dnd-items-v1"},
    )

    print("Fine-tuning job created!")
    print("  Job ID:", job.id)
    print("  Status:", job.status)


if __name__ == "__main__":
    main()
