from src.chunking import chunk_contract
from src.clause_classifier import ClauseClassifier
from src.risk_scorer import calculate_risk, verdict


def test_chunk_overlap():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_contract(text, chunk_size=30, stride=20)
    assert len(chunks) >= 4
    assert set(chunks[0].split()[-10:]) & set(chunks[1].split()[:10])


def test_demo_classifier():
    c = ClauseClassifier(model_dir="/tmp/no-such-model")
    result = c.predict("The agreement shall not be assigned without consent and is governed by the laws of New York.")
    assert "Anti-Assignment" in result["present"]
    assert "Governing Law" in result["present"]


def test_risk_range():
    probs = {"Termination For Convenience": 0.0, "Anti-Assignment": 0.0, "Governing Law": 0.0, "Cap On Liability": 0.0, "Non-Compete": 1.0}
    result = calculate_risk(probs)
    assert 0 <= result["score"] <= 100
    assert result["verdict"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_verdict_boundaries():
    assert verdict(10) == "LOW"
    assert verdict(30) == "MEDIUM"
    assert verdict(60) == "HIGH"
    assert verdict(80) == "CRITICAL"
