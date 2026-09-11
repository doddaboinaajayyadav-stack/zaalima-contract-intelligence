# Implementation Notes

## Mapping to the project plan
- Week 1: OCR/text extraction and baseline entities.
- Week 2: RoBERTa artifact loading and multi-label clause classification.
- Week 3: overlapping chunking, sentence-transformer embeddings, FAISS search, risk scoring, FastAPI.
- Week 4: Docker, Streamlit, pytest, documentation.

## Risk-score interpretation
The project report says Week 3 should blend transformer probabilities (75%) with a classical baseline signal (25%). The report only records the classical model's aggregate F1=0.668, not per-clause probabilities. Therefore this implementation accepts per-clause classical probabilities if later added, and otherwise uses the transformer probability as both signals for a deterministic demo. Do not claim the demo score is a validated legal-risk model until it is calibrated against a labeled validation set.
