from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "roberta_clause_classifier"
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "vector_index"
SAMPLE_DIR = DATA_DIR / "sample_contracts"

CLAUSE_CATEGORIES = [
    "Termination For Convenience",
    "Anti-Assignment",
    "Governing Law",
    "Cap On Liability",
    "Non-Compete",
]

# Risk weights follow the project report's intent. They sum to 100.
ABSENT_RISK = {
    "Termination For Convenience": 18,
    "Anti-Assignment": 18,
    "Governing Law": 10,
    "Cap On Liability": 30,
    "Non-Compete": 0,
}
PRESENT_RISK = {
    "Termination For Convenience": 0,
    "Anti-Assignment": 0,
    "Governing Law": 0,
    "Cap On Liability": 0,
    "Non-Compete": 24,
}

TRANSFORMER_WEIGHT = 0.75
CLASSICAL_WEIGHT = 0.25
