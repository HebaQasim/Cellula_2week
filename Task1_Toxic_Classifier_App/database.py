"""CSV database for Task 2: saves every input with its classification and reads them back."""
from datetime import datetime
from pathlib import Path

import pandas as pd

DATABASE_PATH = Path(__file__).parent / "database.csv"
COLUMNS = ["timestamp", "input_type", "input_text", "classification"]


def save_record(input_type, input_text, classification):
    """Add one row to the CSV file. The file and its header row are created the first time."""
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input_type": input_type,
                "input_text": input_text,
                "classification": classification,
            }
        ],
        columns=COLUMNS,
    )

    file_exists = DATABASE_PATH.exists()
    row.to_csv(DATABASE_PATH, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")


def load_records():
    """Return all saved rows as a table (an empty table if nothing has been saved yet)."""
    if not DATABASE_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(DATABASE_PATH, encoding="utf-8-sig")


if __name__ == "__main__":
    # Runs only with "python database.py": saves two example rows, then shows the whole table
    save_record("text", "You are an idiot, really", "toxic, insult")
    save_record("image", "a dog running on the beach", "non-toxic")
    print(load_records())
