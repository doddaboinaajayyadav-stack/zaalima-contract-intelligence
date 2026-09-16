"""Validate the prepared dataset before model training."""

import json
from pathlib import Path


REQUIRED_FIELDS = {
    "contract_id",
    "question_id",
    "question",
    "context",
    "answer_text",
    "answer_start",
    "category",
    "target",
}


def inspect_jsonl(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            row = json.loads(line)

            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(
                    f"Missing fields in {path} line {line_number}: "
                    f"{sorted(missing)}"
                )

            rows.append(row)

    return rows


def main():
    train = inspect_jsonl("data/processed/train.jsonl")
    val = inspect_jsonl("data/processed/val.jsonl")

    all_rows = train + val
    categories = sorted({row["category"] for row in all_rows})

    train_positive = sum(row["target"] for row in train)
    val_positive = sum(row["target"] for row in val)

    max_context = max(len(row["context"]) for row in all_rows)

    print()
    print("TRAINING DATA CHECK")
    print("-------------------")
    print(f"Train records       : {len(train):,}")
    print(f"Validation records  : {len(val):,}")
    print(f"Total records       : {len(all_rows):,}")
    print(f"Train positives     : {train_positive:,}")
    print(f"Train negatives     : {len(train) - train_positive:,}")
    print(f"Validation positives: {val_positive:,}")
    print(f"Validation negatives: {len(val) - val_positive:,}")
    print(f"Max context chars   : {max_context:,}")

    print()
    print("Categories:")

    for category in categories:
        train_count = sum(
            row["category"] == category for row in train
        )
        val_count = sum(
            row["category"] == category for row in val
        )

        print(
            f"  - {category}: "
            f"train={train_count:,}, val={val_count:,}"
        )

    print()
    print("DATA CHECK PASSED")


if __name__ == "__main__":
    main()
