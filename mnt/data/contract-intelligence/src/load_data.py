"""Flatten CUAD's SQuAD-style JSON into one row per contract/question."""
import json
from pathlib import Path


def load_cuad(input_path: str | Path) -> list[dict]:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    rows = []
    for item in data.get("data", data):
        contract_id = item.get("title", "unknown")
        for para in item.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                answers = qa.get("answers", [])
                rows.append({
                    "contract_id": contract_id,
                    "question_id": qa.get("id", ""),
                    "question": qa.get("question", ""),
                    "context": context,
                    "answer_text": answers[0].get("text", "") if answers else "",
                    "answer_start": answers[0].get("answer_start", -1) if answers else -1,
                })
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cuad/cuad_v1.json")
    parser.add_argument("--output", default="data/processed/cuad_flat.jsonl")
    args = parser.parse_args()
    rows = load_cuad(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):,} rows to {out}")


if __name__ == "__main__":
    main()
