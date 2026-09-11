"""Create a contract-level train/validation split for the five target categories."""
import json
from pathlib import Path
import random

from .config import CLAUSE_CATEGORIES


def prepare(input_path="data/processed/cuad_flat.jsonl", output_dir="data/processed", seed=42):
    rows = [json.loads(x) for x in Path(input_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    # CUAD questions are category labels in natural-language question form.
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
    for r in rows:
        label = category(r.get("question", ""))
        if label:
            selected.append({**r, "category": label, "target": int(bool(r.get("answer_text", "").strip()))})

    contracts = sorted({r["contract_id"] for r in selected})
    rng = random.Random(seed)
    rng.shuffle(contracts)
    split = int(len(contracts) * 0.8)
    train_contracts = set(contracts[:split])
    train = [r for r in selected if r["contract_id"] in train_contracts]
    val = [r for r in selected if r["contract_id"] not in train_contracts]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [("train.jsonl", train), ("val.jsonl", val)]:
        with (out / name).open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"train={len(train):,}, val={len(val):,}, contracts={len(contracts)}")
    return train, val


if __name__ == "__main__":
    prepare()
