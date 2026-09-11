"""Fine-tune a RoBERTa-style encoder on the five binary clause tasks.

For the internship workflow, run this on a GPU (e.g. Kaggle P100), then copy
 the resulting Hugging Face directory to models/roberta_clause_classifier/.
"""
import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def train(train_path="data/processed/train.jsonl", val_path="data/processed/val.jsonl", output="models/roberta_clause_classifier", model_name="roberta-base"):
    train_rows, val_rows = read_jsonl(train_path), read_jsonl(val_path)
    categories = sorted({r["category"] for r in train_rows})
    label2id = {c: i for i, c in enumerate(categories)}
    # Multi-label target: five independent binary labels for each context.
    def group(rows):
        grouped = {}
        for r in rows:
            key = (r["contract_id"], r["context"])
            grouped.setdefault(key, [0] * len(categories))
            grouped[key][label2id[r["category"]]] = r["target"]
        return [{"text": k[1], "labels": v} for k, v in grouped.items()]

    train_ds = Dataset.from_list(group(train_rows))
    val_ds = Dataset.from_list(group(val_rows))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tok(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=512)
        enc["labels"] = batch["labels"]
        return enc

    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(categories), problem_type="multi_label_classification",
        id2label={i: c for c, i in label2id.items()}, label2id=label2id,
    )
    args = TrainingArguments(
        output_dir=output, num_train_epochs=3, per_device_train_batch_size=8,
        per_device_eval_batch_size=8, evaluation_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    train()
