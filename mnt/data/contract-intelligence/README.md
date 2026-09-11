# AI-Powered Contract Intelligence & Risk Scoring

End-to-end NLP project based on the Zaalima Project 1 plan and the Week 1–2 mid-review.

## What is already established
- CUAD is the primary dataset.
- 5 modelling categories: Termination For Convenience, Anti-Assignment, Governing Law, Cap On Liability, Non-Compete.
- Contract-level train/validation split.
- Week 2 reports TF-IDF + Logistic Regression F1=0.668 and fine-tuned RoBERTa F1=0.773.

## Week 3–4 implementation in this repository
1. PDF/DOCX text extraction with OCR fallback.
2. Overlapping contract chunking.
3. Multi-label clause classification using a local Hugging Face model when available.
4. Semantic search with sentence-transformers + FAISS when available, with a TF-IDF fallback.
5. Explainable 0–100 risk scoring and LOW/MEDIUM/HIGH/CRITICAL verdict.
6. FastAPI endpoints: `/health`, `/analyze`, `/classify`, `/search`, `/index`.
7. Streamlit demo UI.
8. Pytest tests and Docker configuration.

## Important model artifact
Put the fine-tuned RoBERTa directory from Kaggle at:
`models/roberta_clause_classifier/`

It should contain the Hugging Face model files (`config.json`, tokenizer files, and model weights). If it is missing, the API still runs in **demo mode** using deterministic rule-based classification, so you can verify the complete pipeline before adding the trained weights.

## 1. Create environment

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Run the API
```bash
uvicorn src.api:app --reload
```

Open Swagger at `http://127.0.0.1:8000/docs`.

## 3. Run the Streamlit UI
In another terminal:
```bash
streamlit run src/streamlit_app.py
```

## 4. Run tests
```bash
pytest -q
```

## 5. Test from the command line
```bash
curl http://127.0.0.1:8000/health
```

For Windows PowerShell, upload a PDF:
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/analyze -Method Post -Form @{file=Get-Item .\data\sample_contracts\sample_contract.txt}
```

Or use Swagger UI for file uploads.

## 6. Build Docker image
```bash
docker build -t contract-intelligence .
docker run -p 8000:8000 contract-intelligence
```

## Suggested final demo flow
PDF/DOCX upload → text extraction/OCR → chunking → clause classification → entity extraction → risk score → semantic search → JSON response → Streamlit visualization.
