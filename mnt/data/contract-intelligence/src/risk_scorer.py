from .config import ABSENT_RISK, PRESENT_RISK, CLAUSE_CATEGORIES


def verdict(score: float) -> str:
    if score < 25:
        return "LOW"
    if score < 50:
        return "MEDIUM"
    if score < 75:
        return "HIGH"
    return "CRITICAL"


def calculate_risk(transformer_probs: dict[str, float], classical_probs: dict[str, float] | None = None) -> dict:
    """Return a transparent 0-100 score.

    The Week 3 plan specifies a 75/25 blend. The available Week 2 report
    gives the classical model's F1, not per-clause probabilities, so this
    implementation uses per-clause classical probabilities when a trained
    classical artifact is available; otherwise transformer probabilities
    are used as the primary signal and the classical component is zero.
    """
    classical_probs = classical_probs or {}
    blended = {}
    contributions = {}
    score = 0.0
    for label in CLAUSE_CATEGORIES:
        tp = float(transformer_probs.get(label, 0.0))
        cp = float(classical_probs.get(label, tp if not classical_probs else 0.0))
        p = 0.75 * tp + 0.25 * cp
        blended[label] = p
        if label == "Non-Compete":
            contribution = p * PRESENT_RISK[label]
        else:
            contribution = (1 - p) * ABSENT_RISK[label]
        contributions[label] = round(contribution, 2)
        score += contribution
    score = max(0.0, min(100.0, score))
    return {
        "score": round(score, 2),
        "verdict": verdict(score),
        "blended_probabilities": blended,
        "risk_contributions": contributions,
    }
