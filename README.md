# D&D Magic Item Generator & Classifiers

This project generates structured Dungeons & Dragons magic items using an OpenAI fine-tuned model and then applies local classifiers (trained on OpenAI embeddings) to predict metadata such as rarity, cost, weight, and tags.

The main entry point is:

- **`generate-and_classify_item.py`**

This script:

1. Accepts a prompt (via CLI)
2. Generates an item using a fine-tuned OpenAI model
3. Runs multiple predictors (rarity, cost, weight, tags)
4. Outputs the final enriched item and saves it to disk

All OpenAI access requires an `.env` file.

---

## Requirements

- Python 3.10+
- OpenAI API key
- `.env` file in the project root
- Dependencies such as:
  - `openai`, `python-dotenv`, `numpy`, `scikit-learn`, `jsonlines`

---

## Environment Setup

Create a `.env` file in the **project root**:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

All scripts reference this key using:

```python
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

If this variable is missing, _all API calls will fail_.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

(If you don’t have a requirements file, install dependencies manually.)

---

## Project Structure

```
project/
│
├── generate-and_classify_item.py         # Entry point
├── .env                                  # OpenAI key (not committed)
│
├── generated-classifiers/                # Saved classifier .pkl files
│   ├── rarity_classifier.pkl
│   ├── cost_regressor.pkl
│   └── ...
│
├── predictors/                           # Predictors that use the .pkl models
│   ├── cost_predictor.py
│   ├── rarity_classifier.py
│   └── ...
│
├── item-field-classifier-generation/     # Scripts that train classifiers
│   ├── rarity_dataset.jsonl
│   ├── train_rarity_classifier.py
│   └── ...
│
├── openai-model-finetuning/              # Scripts for data prep & fine-tuning
│   ├── prepare_training_data.py
│   └── ...
```
