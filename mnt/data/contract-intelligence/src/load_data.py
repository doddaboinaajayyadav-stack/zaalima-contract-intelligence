"""Create a contract-level train/validation split for the five target categories."""
import json
from pathlib import Path
import random


def prepare(
    input_path="data/processed/cuad_flat.jsonl",
    output_dir="data/processed",
    seed=42,
):
    input_path = Path(input_path)

    rows = []

    # Read JSONL safely line-by-line.
    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

    def category(question: str):
        q = question.lower()

        mapping = {
            "termination for convenience": "Termination For Convenience",
            "anti-assignment": "Anti-Assignment",
            "governing law": "Governing Law",
            "cap on liability": "Cap On Liability",
            "non-compete": "Non-Compete",
        }

        for needle, label in mapping.items():
            if needle in q:
                return label

        return None

    selected = []

    for row in rows:
        label = category(row.get("question", ""))

        if label:
            selected.append(
                {
                    **row,
                    "category": label,
                    "target": int(
                        bool(row.get("answer_text", "").strip())
                    ),
                }
            )

    contracts = sorted(
        {row["contract_id"] for row in selected}
    )

    rng = random.Random(seed)
    rng.shuffle(contracts)

    split = int(len(contracts) * 0.8)

    train_contracts = set(contracts[:split])

    train = [
        row
        for row in selected
        if row["contract_id"] in train_contracts
    ]

    val = [
        row
        for row in selected
        if row["contract_id"] not in train_contracts
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [
        ("train.jsonl", train),
        ("val.jsonl", val),
    ]:
        output_path = output_dir / name

        with output_path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(
        f"train={len(train):,}, "
        f"val={len(val):,}, "
        f"contracts={len(contracts)}"
    )

    return train, val


if __name__ == "__main__":
    prepare()