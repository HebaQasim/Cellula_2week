# Task 2 — Toxic Content Classification App

A Streamlit app that checks a typed comment or an image for toxic content and saves every input and its result in a CSV database.

## How it works

```text
Typed text ──────────────────────────────────┐
                                             ├──► BiLSTM classifier (Task 1) ──► result ──► database.csv
Image ──► BLIP-1 caption (a short sentence) ─┘
```

- **Image captioning:** `Salesforce/blip-image-captioning-base` (BLIP-1). It turns an image into a short English description. The model is downloaded from Hugging Face the first time it is used (about 1 GB) and cached afterwards.
- **Text classification:** the BiLSTM from Task 1, trained on the Jigsaw Toxic Comment dataset. It gives one probability for each of six labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. Each label has its own decision threshold, tuned on the validation set in Task 1. The result is the list of labels whose probability reaches their threshold, or `non-toxic` if none does.
- **Database:** `database.csv`, created automatically. One row is added for every submitted text or image caption.

| Column | Content |
|---|---|
| `timestamp` | When the input was submitted |
| `input_type` | `text` or `image` |
| `input_text` | The typed text, or the caption generated for the image |
| `classification` | The result, e.g. `toxic, insult` or `non-toxic` |

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app (main script): Text, Image and Database tabs |
| `imagecaption.py` | `generate_caption(image)`: image → caption with BLIP-1 |
| `classifier.py` | `classify_text(text)`: text → six probabilities → labels, using the Task 1 model |
| `database.py` | `save_record(...)` adds a row to the CSV; `load_records()` reads all rows |
| `models/lstm_model.keras` | Trained Task 1 BiLSTM |
| `models/tokenizer.json` | Task 1 tokenizer (fitted on the training set, 20,000-word vocabulary) |
| `models/lstm_thresholds.csv` | Per-label thresholds tuned on the Task 1 validation set |
| `requirements.txt` | Python packages and versions |

## Setup and run (Windows)

Requires **Python 3.10–3.13** (TensorFlow 2.20 does not support Python 3.14).

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens in the browser at `http://localhost:8501`.

## Using the app

- **Text tab:** write a comment and press *Classify text*.
- **Image tab:** upload a JPG or PNG image and press *Classify image*. The caption is shown, then classified.
- **Database tab:** shows every stored input with its classification.

Each module can also be tested on its own: `python classifier.py`, `python imagecaption.py` (needs a `test_image.jpg` next to it) and `python database.py`.

## Notes and limitations

- New text goes through exactly the same preprocessing as the Task 1 training data: the same `clean_text` function, the saved tokenizer, length 250 and pre-padding.
- An input with no words at all (for example `===`) is not sent to the model; it is stored as `no words to classify`.
- The classifier learned what toxic **language** looks like in Wikipedia comments. BLIP captions are short, neutral descriptions (for example "a man holding a knife"), so images are usually classified as `non-toxic`, even when the scene itself is violent.
