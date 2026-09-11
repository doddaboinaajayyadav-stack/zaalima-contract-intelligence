"""Train one TF-IDF + Logistic Regression classifier per clause category."""
import json
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def train(train_path="data/processed/train.jsonl", val_path="data/processed/val.jsonl", out_dir="models/classical"):
    train_rows, val_rows = read_jsonl(train_path), read_jsonl(val_path)
    categories = sorted({r["category"] for r in train_rows})
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for cat in categories:
        tr = [r for r in train_rows if r["category"] == cat]
        va = [r for r in val_rows if r["category"] == cat]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        Xtr = vectorizer.fit_transform([r["context"] for r in tr])
        Xva = vectorizer.transform([r["context"] for r in va])
        ytr = [r["target"] for r in tr]
        yva = [r["target"] for r in va]
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        results[cat] = {
            "accuracy": accuracy_score(yva, pred),
            "precision": precision_score(yva, pred, zero_division=0),
            "recall": recall_score(yva, pred, zero_division=0),
            "f1": f1_score(yva, pred, zero_division=0),
        }
        joblib.dump((vectorizer, model), out / f"{cat.replace(' ', '_').lower()}.joblib")
    avg = {m: sum(v[m] for v in results.values()) / len(results) for m in ["accuracy", "precision", "recall", "f1"]}
    (out / "results.json").write_text(json.dumps({"per_category": results, "average": avg}, indent=2), encoding="utf-8")
    print(json.dumps(avg, indent=2))


if __name__ == "__main__":
    train()
