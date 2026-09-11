from pathlib import Path
import pickle
import numpy as np

from .config import INDEX_DIR


class VectorStore:
    """FAISS + MiniLM semantic search, with a lightweight TF-IDF fallback."""
    def __init__(self, index_dir: str | Path = INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.model = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.mode = "none"
        self.chunks: list[str] = []
        self._load()

    def _load(self):
        meta = self.index_dir / "chunks.pkl"
        faiss_file = self.index_dir / "index.faiss"
        tfidf_file = self.index_dir / "tfidf.pkl"
        if meta.exists():
            with open(meta, "rb") as f:
                self.chunks = pickle.load(f)
        if faiss_file.exists():
            try:
                import faiss
                from sentence_transformers import SentenceTransformer
                self.index = faiss.read_index(str(faiss_file))
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.mode = "faiss-minilm"
                return
            except Exception:
                pass
        if tfidf_file.exists():
            try:
                with open(tfidf_file, "rb") as f:
                    self.vectorizer, self.tfidf_matrix = pickle.load(f)
                self.mode = "tfidf"
            except Exception:
                pass

    def build(self, chunks: list[str], prefer_semantic: bool = True):
        if not chunks:
            raise ValueError("No chunks supplied")
        self.chunks = chunks
        if prefer_semantic:
            try:
                from sentence_transformers import SentenceTransformer
                import faiss
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = self.model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
                embeddings = np.asarray(embeddings, dtype="float32")
                self.index = faiss.IndexFlatIP(embeddings.shape[1])
                self.index.add(embeddings)
                faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
                with open(self.index_dir / "chunks.pkl", "wb") as f:
                    pickle.dump(chunks, f)
                self.mode = "faiss-minilm"
                return
            except Exception as exc:
                print(f"Semantic index unavailable; using TF-IDF fallback: {exc}")

        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)
        with open(self.index_dir / "tfidf.pkl", "wb") as f:
            pickle.dump((self.vectorizer, self.tfidf_matrix), f)
        with open(self.index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        self.mode = "tfidf"

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self.chunks:
            return []
        if self.mode == "faiss-minilm" and self.index is not None and self.model is not None:
            q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            q = np.asarray(q, dtype="float32")
            scores, ids = self.index.search(q, min(k, len(self.chunks)))
            return [
                {"chunk": self.chunks[int(i)], "score": round(float(s), 4)}
                for s, i in zip(scores[0], ids[0]) if int(i) >= 0
            ]
        if self.mode == "tfidf" and self.vectorizer is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            q = self.vectorizer.transform([query])
            scores = cosine_similarity(q, self.tfidf_matrix)[0]
            ids = np.argsort(scores)[::-1][:k]
            return [
                {"chunk": self.chunks[int(i)], "score": round(float(scores[i]), 4)}
                for i in ids if scores[i] > 0
            ]
        return []
