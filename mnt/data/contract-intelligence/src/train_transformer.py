"""Fine-tune a RoBERTa-style encoder on the five binary clause tasks.

Run training on a GPU such as Kaggle.
"""

import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


def read_jsonl(path):
    rows = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def train(
    train_path="data/processed/train.jsonl",
    val_path="data/processed/val.jsonl",
    output="models/roberta_clause_classifier",
    model_name="roberta-base",
):

    print("Loading training data...")

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)

    print(f"Training rows: {len(train_rows):,}")
    print(f"Validation rows: {len(val_rows):,}")

    categories = sorted(
        {row["category"] for row in train_rows}
    )

    label2id = {
        category: index
        for index, category in enumerate(categories)
    }

    id2label = {
        index: category
        for category, index in label2id.items()
    }

    print("Categories:")
    for category in categories:
        print(f"  {category}")

    def group(rows):

        grouped = {}

        for row in rows:

            key = (
                row["contract_id"],
                row["context"],
            )

            if key not in grouped:
                grouped[key] = [0] * len(categories)

            grouped[key][
                label2id[row["category"]]
            ] = row["target"]

        return [
            {
                "text": context,
                "labels": labels,
            }
            for (_, context), labels in grouped.items()
        ]

    train_data = group(train_rows)
    val_data = group(val_rows)

    print(f"Grouped training examples: {len(train_data):,}")
    print(f"Grouped validation examples: {len(val_data):,}")

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):

        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
        )

        encoded["labels"] = batch["labels"]

        return encoded

    print("Tokenizing training data...")

    train_ds = train_ds.map(
        tokenize,
        batched=True,
    )

    print("Tokenizing validation data...")

    val_ds = val_ds.map(
        tokenize,
        batched=True,
    )

    print("Loading RoBERTa model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(categories),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=output,

        num_train_epochs=3,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,

        gradient_accumulation_steps=2,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        logging_steps=50,

        save_total_limit=2,

        report_to="none",

        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("Starting training...")

    trainer.train()

    print("Saving model...")

    trainer.save_model(output)
    tokenizer.save_pretrained(output)

    print(f"Saved model to {output}")


if __name__ == "__main__":
    train()