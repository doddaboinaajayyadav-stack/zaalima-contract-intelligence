from pathlib import Path
import re
from typing import Any

from .config import CLAUSE_CATEGORIES, MODEL_DIR

KEYWORDS = {
    "Termination For Convenience": ["terminate for convenience", "termination for convenience", "may terminate without cause"],
    "Anti-Assignment": ["may not assign", "shall not assign", "shall not be assigned", "may not be assigned", "assignment", "transfer this agreement"],
    "Governing Law": ["governed by the laws", "governing law", "laws of the state", "jurisdiction"],
    "Cap On Liability": ["limitation of liability", "cap on liability", "aggregate liability", "liability shall not exceed"],
    "Non-Compete": ["non-compete", "noncompete", "shall not compete", "refrain from competing", "compete with"],
}


class ClauseClassifier:
    def __init__(self, model_dir: str | Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.pipeline = None
        self.mode = "demo"
        self._load_model_if_available()

    def _load_model_if_available(self):
        config = self.model_dir / "config.json"
        if not config.exists():
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                top_k=None,
                truncation=True,
                max_length=512,
            )
            self.mode = "roberta"
        except Exception as exc:
            print(f"Model load failed; using demo classifier: {exc}")

    @staticmethod
    def _sigmoid(x: float) -> float:
        import math
        return 1 / (1 + math.exp(-x))

    def _demo_predict(self, text: str) -> dict[str, float]:
        lower = text.lower()
        scores = {}
        for label in CLAUSE_CATEGORIES:
            hits = sum(lower.count(k) for k in KEYWORDS[label])
            scores[label] = min(0.98, 0.08 + 0.55 * hits)
        return scores

    def _roberta_predict(self, text: str) -> dict[str, float]:
        raw = self.pipeline(text)
        items = raw[0] if raw and isinstance(raw[0], list) else raw
        out = {label: 0.0 for label in CLAUSE_CATEGORIES}
        for item in items:
            label = str(item["label"])
            score = float(item["score"])
            # Support models saved with labels like LABEL_0 through id2label.
            if label in out:
                out[label] = score
        # If the model has numeric labels, map by category order.
        if max(out.values(), default=0.0) == 0.0:
            for item in items:
                m = re.search(r"(\d+)$", str(item["label"]))
                if m:
                    idx = int(m.group(1))
                    if idx < len(CLAUSE_CATEGORIES):
                        out[CLAUSE_CATEGORIES[idx]] = float(item["score"])
        return out

    def predict(self, text: str) -> dict[str, Any]:
        probs = self._roberta_predict(text) if self.pipeline else self._demo_predict(text)
        return {
            "model": self.mode,
            "probabilities": probs,
            "present": [k for k, v in probs.items() if v >= 0.5],
        }
