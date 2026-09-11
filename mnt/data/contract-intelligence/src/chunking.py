def chunk_contract(text: str, chunk_size: int = 350, stride: int = 280) -> list[str]:
    """Overlapping word-level chunks, as specified for Week 3."""
    words = text.split()
    if not words:
        return []
    if stride <= 0 or chunk_size <= 0 or stride > chunk_size:
        raise ValueError("Require 0 < stride <= chunk_size")
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
        start += stride
    return chunks
