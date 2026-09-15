# AI-Powered Contract Intelligence & Risk Scoring

An NLP-powered contract analysis platform designed to assist legal and compliance teams in reviewing commercial contracts.

The system accepts PDF, DOCX, TXT, and Markdown contracts, extracts document text, identifies important entities, classifies legal clauses, calculates a transparent risk score, creates searchable document chunks, and exposes the workflow through a FastAPI backend and Streamlit dashboard.

---

## 📌 Project Overview

Legal contracts can contain hundreds of pages of dense and complex language. Manually reviewing every clause is time-consuming and increases the possibility of missing unfavorable or high-risk terms.

This project provides an automated contract intelligence pipeline that helps users:

- Upload contracts
- Extract text from documents
- Apply OCR fallback for scanned PDFs
- Identify dates, organizations, monetary values, and jurisdictions
- Detect important legal clauses
- Calculate an overall contract risk score
- Categorize risk as LOW, MEDIUM, HIGH, or CRITICAL
- Split contracts into searchable chunks
- Build a vector search index
- Query the FastAPI backend
- Visualize results through a Streamlit dashboard

---

## 🎯 Objectives

The main objectives are:

1. Automate initial contract review.
2. Extract important information from legal documents.
3. Identify potentially risky contractual clauses.
4. Provide an interpretable risk score.
5. Enable semantic search over contract content.
6. Provide REST APIs for integration.
7. Provide a lightweight user interface.
8. Build a modular architecture that can later support production-grade ML models.

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │        User          │
                    │ Legal / Compliance   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Dashboard  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      FastAPI API     │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
       │ PDF / DOCX   │ │ NLP Pipeline │ │ Vector Search│
       │ Extraction   │ │              │ │              │
       │ + OCR        │ │ Entities     │ │ Embeddings   │
       └──────┬───────┘ │ Clauses      │ │ FAISS        │
              │         │ Risk         │ └──────┬───────┘
              │         └──────┬───────┘        │
              └─────────────────┼────────────────┘
                                ▼
                       ┌─────────────────┐
                       │ Analysis Result │
                       └─────────────────┘

  ## 🔄 Processing Pipeline

The application follows an end-to-end contract analysis workflow:

1. **Contract Upload**
   - Accepts PDF, DOCX, TXT, and MD files.

2. **Text Extraction**
   - Extracts text from digital documents.
   - Uses OCR as a fallback for scanned PDF pages.

3. **Contract Chunking**
   - Splits extracted contract text into overlapping chunks for NLP and semantic search.

4. **Clause Classification**
   - Detects important contractual clauses including:
     - Termination For Convenience
     - Anti-Assignment
     - Governing Law
     - Cap On Liability
     - Non-Compete

5. **Entity Extraction**
   - Identifies organizations, dates, monetary values, and jurisdictions.

6. **Risk Scoring**
   - Generates a transparent 0–100 risk score.
   - Classifies contracts as:
     - LOW
     - MEDIUM
     - HIGH
     - CRITICAL

7. **Vector Indexing**
   - Creates vector representations of contract chunks.
   - Stores them using FAISS for semantic search.

8. **Analysis Result**
   - Combines extracted text, detected clauses, entities, risk assessment, and vector indexing into the final analysis response.

9. **Streamlit Dashboard**
   - Presents the analysis through an interactive web interface.                     


   ## 🚀 Features

- 📄 **Multi-format contract ingestion**
  - PDF
  - DOCX
  - TXT
  - Markdown

- 🔍 **OCR fallback**
  - Handles scanned and image-based PDF documents.

- 🤖 **Clause classification**
  - Identifies key contractual clauses using the NLP classification pipeline.

- 🏢 **Entity extraction**
  - Extracts organizations, dates, monetary values, and jurisdictions.

- ⚠️ **Risk assessment**
  - Produces a transparent 0–100 risk score.
  - Provides LOW, MEDIUM, HIGH, and CRITICAL verdicts.

- 🧩 **Contract chunking**
  - Splits large contracts into overlapping text chunks for analysis.

- 🔎 **Semantic search**
  - Uses embeddings and FAISS for contract-level similarity search.

- ⚡ **FastAPI backend**
  - Provides REST endpoints for health checks, classification, indexing, searching, and analysis.

- 📊 **Streamlit dashboard**
  - Provides an interactive interface for uploading and analyzing contracts.

- 🧪 **Automated testing**
  - Includes pytest-based tests for core functionality.



 ## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **NLP / ML** | Python, Transformers, Sentence Transformers, spaCy, scikit-learn |
| **Deep Learning** | PyTorch |
| **Document Processing** | pdfplumber, python-docx, Pillow, Tesseract OCR |
| **Vector Search** | FAISS |
| **Data Processing** | NumPy, Pandas |
| **API Validation** | Pydantic |
| **Testing** | pytest |
| **Containerization** | Docker |
| **Version Control** | Git, GitHub |

## 📁 Project Structure

```text
contract-intelligence/
│
├── data/
│   ├── sample_contracts/
│   ├── cuad/
│   ├── processed/
│   └── vector_index/
│
├── docs/
│
├── models/
│   └── roberta_clause_classifier/
│
├── src/
│   ├── api.py
│   ├── chunking.py
│   ├── clause_classifier.py
│   ├── config.py
│   ├── entities.py
│   ├── load_data.py
│   ├── ocr_pipeline.py
│   ├── prepare_training_data.py
│   ├── risk_scorer.py
│   ├── streamlit_app.py
│   ├── train_classical_baseline.py
│   ├── train_transformer.py
│   └── vector_store.py
│
├── tests/
│
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/doddaboinaajayyadav-stack/zaalima-contract-intelligence.git
cd zaalima-contract-intelligence
```

### 2. Create a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Start the FastAPI Backend

Run from the project directory:

```powershell
python -m uvicorn src.api:app --port 8001
```

The API will be available at:

```text
http://127.0.0.1:8001
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

### 5. Start the Streamlit Dashboard

Open a second terminal, activate the virtual environment, navigate to the project directory, and run:

```powershell
python -m streamlit run src\streamlit_app.py --server.port 8501
```

The dashboard will be available at:

```text
http://localhost:8501
```

### 6. Run Tests

From the project directory:

```powershell
python -m pytest -q
```

Expected result:

```text
4 passed
```


## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check API and model status |
| `POST` | `/classify` | Classify contract text into clause categories |
| `POST` | `/analyze` | Upload and analyze a complete contract |
| `POST` | `/index` | Build a vector index from contract text |
| `POST` | `/search` | Perform semantic search over indexed contract chunks |

### Example Health Check

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

### Example Contract Analysis

```powershell
curl.exe -X POST "http://127.0.0.1:8001/analyze" -F "file=@data\sample_contracts\EMPLOYMENT-AGREEMENT.pdf"
```

The `/analyze` endpoint returns:

- Extracted text statistics
- Contract chunks
- Detected entities
- Clause classification
- Clause probabilities
- Risk score
- Risk verdict
- Vector indexing status

## 🧪 Testing

The project includes automated tests using pytest.

Run:

```powershell
python -m pytest -q
```

Current test result:

```text
4 passed
```

## 📌 Project Status

The current implementation provides an end-to-end contract intelligence pipeline covering:

- Document ingestion
- Text extraction
- OCR fallback
- Contract chunking
- Clause classification
- Entity extraction
- Risk scoring
- Vector indexing
- Semantic search
- FastAPI backend
- Streamlit dashboard
- Automated testing

Further improvements can include training and integrating production-grade classification models, improving entity extraction accuracy, expanding clause categories, and adding more comprehensive test coverage.


## 🧠 ML Training Pipeline

The project includes a complete data preparation pipeline for training a
multi-label RoBERTa-based clause classifier.

### Dataset

The system uses the CUAD (Contract Understanding Atticus Dataset) for
contract clause classification.

Current dataset preparation:

- 510 contracts
- 20,910 flattened contract/question records
- 2,448 training records
- 612 validation records
- 80/20 contract-level train/validation split
- Five target clause categories:
  - Termination For Convenience
  - Anti-Assignment
  - Governing Law
  - Cap On Liability
  - Non-Compete

### Training Workflow

```text
CUAD Dataset
     ↓
CUADv1.json
     ↓
load_data.py
     ↓
cuad_flat.jsonl
     ↓
prepare_training_data.py
     ↓
train.jsonl + val.jsonl
     ↓
train_transformer.py
     ↓
RoBERTa Clause Classifier
     ↓
models/roberta_clause_classifier/