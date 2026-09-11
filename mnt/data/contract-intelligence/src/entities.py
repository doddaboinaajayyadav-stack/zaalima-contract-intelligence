import re


def extract_entities(text: str) -> dict:
    # Lightweight baseline, consistent with the Week 1 proof-of-concept.
    money = re.findall(r"(?:USD|INR|EUR|GBP|\$|€|£)\s?[\d,]+(?:\.\d+)?", text, flags=re.I)
    dates = re.findall(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
        text,
    )
    jurisdictions = re.findall(
        r"(?:laws? of|courts? of|jurisdiction of)\s+([A-Z][A-Za-z .,&-]{2,60})",
        text,
    )
    orgs = re.findall(
        r"\b([A-Z][A-Za-z0-9&.,' -]{2,50}\s(?:Inc\.|LLC|Ltd\.|Limited|Corporation|Corp\.))\b",
        text,
    )
    return {
        "organizations": sorted(set(orgs))[:50],
        "dates": sorted(set(dates))[:50],
        "monetary_values": sorted(set(money))[:50],
        "jurisdictions": sorted(set(j.strip() for j in jurisdictions))[:50],
    }
