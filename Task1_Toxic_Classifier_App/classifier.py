"""Text classifier for Task 2: uses the Task 1 BiLSTM to predict the six toxicity labels."""
from functools import lru_cache
from pathlib import Path

import keras
import pandas as pd
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# ---------- File locations: the "models" folder next to this file ----------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "lstm_model.keras"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.json"
THRESHOLDS_PATH = BASE_DIR / "models" / "lstm_thresholds.csv"

# ---------- Settings copied from the Task 1 notebook ----------
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_SEQUENCE_LENGTH = 250
URL_REPLACE_PATTERN = r"https?://\S+|\bwww\.\S+"
IP_PATTERN = r"\b\d{1,3}(?:\.\d{1,3}){3}\b"
CONTROL_CHAR_PATTERN = r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"


def clean_text(texts):
    """Same cleaning as the notebook (Section 11)."""
    return (
        texts.str.lower()
        .str.replace(URL_REPLACE_PATTERN, " urltoken ", regex=True)
        .str.replace(IP_PATTERN, " iptoken ", regex=True)
        .str.replace(CONTROL_CHAR_PATTERN, " ", regex=True)
        .str.split().str.join(" ")
    )


def masked_global_max_pooling(sequence_outputs, mask):
    """Same function as the notebook: max over real words only (padding is ignored)."""
    mask = keras.ops.expand_dims(keras.ops.cast(mask, sequence_outputs.dtype), axis=-1)
    return keras.ops.max(sequence_outputs - (1.0 - mask) * 1e9, axis=1)


@lru_cache(maxsize=1)
def load_classifier():
    """Load the model, tokenizer and thresholds the first time; later calls reuse them."""
    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"masked_global_max_pooling": masked_global_max_pooling},
    )
    bilstm = model.get_layer("bilstm")
    bilstm.forward_layer.use_cudnn = False
    bilstm.backward_layer.use_cudnn = False

    with open(TOKENIZER_PATH, encoding="utf-8") as file:
        tokenizer = tokenizer_from_json(file.read())

    threshold_table = pd.read_csv(THRESHOLDS_PATH, index_col="label")
    thresholds = threshold_table.loc[LABELS, "tuned_threshold"].to_numpy()

    return model, tokenizer, thresholds


def classify_text(text):
    """Classify one text and return the result, the predicted labels and the six probabilities."""
    model, tokenizer, thresholds = load_classifier()

    # Step 3: clean -> numbers -> pad
    cleaned = clean_text(pd.Series([text]))
    sequences = tokenizer.texts_to_sequences(cleaned)

    if len(sequences[0]) == 0:
        return {"result": "no words to classify", "labels": [], "probabilities": {}}

    padded = keras.utils.pad_sequences(
        sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="pre"
    )

    # Step 4: predict -> compare with thresholds
    probabilities = model.predict(padded, verbose=0)[0]

    predicted_labels = []
    probability_by_label = {}
    for i in range(len(LABELS)):
        probability_by_label[LABELS[i]] = round(float(probabilities[i]), 4)
        if probabilities[i] >= thresholds[i]:
            predicted_labels.append(LABELS[i])

    if len(predicted_labels) > 0:
        result = ", ".join(predicted_labels)
    else:
        result = "non-toxic"

    return {"result": result, "labels": predicted_labels, "probabilities": probability_by_label}


if __name__ == "__main__":
    # Runs only with "python classifier.py", not when another file imports this one
    examples = [
        "Thank you for fixing the article",
        "You are an IDIOT!! http://x.com",
        "I will find you and kill you",
        "===",
    ]
    for example in examples:
        print(example)
        print("   ", classify_text(example))
