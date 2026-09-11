from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .clause_classifier import ClauseClassifier
from .chunking import chunk_contract
from .entities import extract_entities
from .ocr_pipeline import extract_text
from .risk_scorer import calculate_risk
from .vector_store import VectorStore

app = FastAPI(title="Contract Intelligence API", version="1.0.0")
classifier = ClauseClassifier()
vector_store = VectorStore()


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=20)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    k: int = Field(default=5, ge=1, le=20)


class IndexRequest(BaseModel):
    text: str = Field(min_length=20)
    chunk_size: int = Field(default=350, ge=50, le=1000)
    stride: int = Field(default=280, ge=25, le=1000)


@app.get("/health")
def health():
    return {"status": "ok", "classifier_mode": classifier.mode, "vector_indexed": vector_store.index is not None}


@app.post("/classify")
def classify(req: ClassifyRequest):
    result = classifier.predict(req.text)
    return result


@app.post("/index")
def index(req: IndexRequest):
    chunks = chunk_contract(req.text, req.chunk_size, req.stride)
    try:
        vector_store.build(chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vector indexing failed: {exc}")
    return {"indexed_chunks": len(chunks)}


@app.post("/search")
def search(req: SearchRequest):
    return {"query": req.query, "results": vector_store.search(req.query, req.k)}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "contract.txt").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, TXT and MD files are supported")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        text = extract_text(tmp_path)
        if len(text.strip()) < 20:
            raise HTTPException(status_code=422, detail="Could not extract enough text from the document")
        classification = classifier.predict(text[:12000])
        risk = calculate_risk(classification["probabilities"])
        chunks = chunk_contract(text)
        # Build the search index as part of the demo pipeline.
        try:
            vector_store.build(chunks)
            indexed = True
        except Exception:
            indexed = False
        return {
            "filename": file.filename,
            "text_characters": len(text),
            "chunks": len(chunks),
            "entities": extract_entities(text),
            "classification": classification,
            "risk": risk,
            "vector_indexed": indexed,
        }
    finally:
        tmp_path.unlink(missing_ok=True)
