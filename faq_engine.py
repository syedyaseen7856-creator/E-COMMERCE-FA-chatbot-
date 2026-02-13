import math
import textwrap
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


class FAQEngine:
    """Encapsulates loading, summarizing, embedding and searching FAQ data.

    Workflow:
    - load JSON data containing products and policies
    - compress long text using a summarization model
    - embed summaries with SentenceTransformers
    - store embeddings + metadata in a FAISS index for semantic search
    """

    def __init__(self, data: List[Dict[str, Any]],
                 summarizer_model: str = "facebook/bart-large-cnn",
                 embed_model: str = "all-MiniLM-L6-v2"):
        self.raw_data = data
        self.summarizer_model_name = summarizer_model
        self.embed_model_name = embed_model

        # Will be initialized in setup
        self.summarizer = None
        self.embedder = None
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension: int = 0

        # Prepared compressed texts along with mapping to metadata
        self._summaries: List[str] = []

        # Initialize models and index
        self._setup()

    def _setup(self):
        # Load summarization pipeline (CPU-friendly by default)
        self.summarizer = pipeline("summarization", model=self.summarizer_model_name, device=-1)

        # Load sentence-transformers model for embeddings
        self.embedder = SentenceTransformer(self.embed_model_name)
        self.dimension = self.embedder.get_sentence_embedding_dimension()

        # Process dataset: summarize -> embed -> build FAISS
        self._prepare_corpus()
        self._build_faiss_index()

    # ---------------------------- Data preparation ----------------------------
    def _chunk_text(self, text: str, max_chars: int = 900) -> List[str]:
        """Simple heuristic chunker to keep summarizer within model limits.

        Splits by paragraphs and falls back to fixed-size chunks.
        """
        # Prefer splitting on paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) + 2 <= max_chars:
                current = (current + "\n\n" + p).strip() if current else p
            else:
                if current:
                    chunks.append(current)
                if len(p) <= max_chars:
                    current = p
                else:
                    # hard split
                    for i in range(0, len(p), max_chars):
                        chunks.append(p[i:i + max_chars])
                    current = ""
        if current:
            chunks.append(current)
        return chunks

    def _summarize_text(self, text: str) -> str:
        """Summarize long text safely by chunking and then (optionally) compressing again.

        Returns a concise string representation suitable for embedding/search.
        """
        if not text:
            return ""

        # If short, return as-is
        if len(text) < 800:
            return text.strip()

        chunks = self._chunk_text(text, max_chars=900)
        summaries = []
        for c in chunks:
            # Use summarizer pipeline with conservative lengths
            try:
                out = self.summarizer(c, max_length=130, min_length=30, do_sample=False)
                summaries.append(out[0]["summary_text"].strip())
            except Exception:
                # Fallback to the raw chunk if summarization fails
                summaries.append(c)

        joined = " \n\n ".join(summaries)

        # If the concatenated summary is still long, summarize again
        if len(joined) > 1000:
            try:
                out = self.summarizer(joined, max_length=180, min_length=40, do_sample=False)
                return out[0]["summary_text"].strip()
            except Exception:
                return joined
        return joined

    def _prepare_corpus(self):
        """Summarize all items in the raw dataset and build metadata list."""
        self._summaries = []
        self.metadata = []

        for item in self.raw_data:
            content = item.get("content", "")
            summary = self._summarize_text(content)
            self._summaries.append(summary)

            # Record metadata that will be returned with search results
            self.metadata.append({
                "id": item.get("id"),
                "title": item.get("title"),
                "category": item.get("category"),
                "source": item.get("source", "dataset"),
                "original": content,
            })

    # ---------------------------- FAISS handling -----------------------------
    def _build_faiss_index(self):
        """Create a FAISS index from the summaries.

        We use cosine-similarity via normalized vectors + IndexFlatIP.
        """
        if not self._summaries:
            raise ValueError("No summaries available to build index.")

        # Encode summaries to embeddings
        embeddings = self.embedder.encode(self._summaries, convert_to_numpy=True, show_progress_bar=False)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create FAISS index and add embeddings
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(np.array(embeddings, dtype="float32"))

    # ---------------------------- Query API --------------------------------
    def query(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Run semantic search over the FAISS index and return top_k results.

        Returns list of dicts with: summary (answer), metadata, score (cosine similarity)
        """
        if not question or not question.strip():
            return []

        q_emb = self.embedder.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        scores, idxs = self.index.search(np.array(q_emb, dtype="float32"), top_k)
        results: List[Dict[str, Any]] = []

        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            # Cosine similarity score is in [-1, 1]; clamp to [0,1] for confidence
            conf = float(max(min(score, 1.0), -1.0))
            conf = max(0.0, conf)

            results.append({
                "answer": self._summaries[idx],
                "metadata": self.metadata[idx],
                "score": conf,
            })
        return results


# ------------------------------- Utility ----------------------------------
if __name__ == "__main__":
    # Quick local test (not run as part of package)
    sample = [
        {"id": 1, "title": "Test product", "category": "product", "content": "This is a long product description..."}
    ]
    engine = FAQEngine(sample)
    print(engine.query("shipping times"))
