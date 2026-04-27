"""
vector_store.py — Step 2: Embedding & Vector Store
===================================================
Embeds text chunks using a sentence-transformer model
and stores them in a FAISS index for fast similarity search.

Install:
    pip install sentence-transformers faiss-cpu numpy --break-system-packages
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os


class VectorStore:
    """
    Manages text embeddings and similarity search via FAISS.

    Attributes:
        model_name: Sentence-transformer model for encoding text.
        model:      Loaded SentenceTransformer instance.
        index:      FAISS index for fast nearest-neighbour search.
        chunks:     List of original text chunks (parallel to index).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialise the vector store.

        Args:
            model_name: HuggingFace model name. 'all-MiniLM-L6-v2' is fast
                        and accurate; swap for 'all-mpnet-base-v2' for higher
                        quality at the cost of speed.
        """
        print(f"🔄  Loading embedding model: {model_name} …")
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.chunks: list[str] = []

    # ── Building the index ───────────────────────────────────────

    def build_index(self, chunks: list[str]) -> None:
        """
        Encode chunks and build the FAISS index.

        Args:
            chunks: List of text strings to embed and store.
        """
        if not chunks:
            raise ValueError("chunks list is empty — nothing to index.")

        self.chunks = chunks
        print(f"⚙️   Embedding {len(chunks)} chunks …")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")

        # Normalise for cosine similarity via inner-product search
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        # IndexFlatIP = exact inner-product (cosine after normalisation)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"✅  Index built: {self.index.ntotal} vectors, dim={dimension}")

    # ── Searching ────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """
        Return the top-k most relevant chunks for a query.

        Args:
            query: User's natural-language query.
            top_k: Number of chunks to retrieve.

        Returns:
            List of text chunks ranked by relevance.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() or load() first.")

        query_vec = self.model.encode([query], dtype="float32")
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # -1 means no result found
                results.append(self.chunks[idx])

        return results

    # ── Persistence ──────────────────────────────────────────────

    def save(self, path: str = "vector_store") -> None:
        """
        Persist the FAISS index and chunk list to disk.

        Args:
            path: Base file path (no extension). Creates <path>.index
                  and <path>.pkl.
        """
        if self.index is None:
            raise RuntimeError("No index to save.")

        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"💾  Saved index to {path}.index + {path}.pkl")

    def load(self, path: str = "vector_store") -> None:
        """
        Load a previously saved index from disk.

        Args:
            path: Base file path used when saving.
        """
        index_file = f"{path}.index"
        chunks_file = f"{path}.pkl"

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

        self.index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"📂  Loaded index: {self.index.ntotal} vectors")


# ── Example usage ────────────────────────────────────────────────
if __name__ == "__main__":
    from ingest_pdf import ingest_multiple_pdfs

    # Build and save index (run once)
    pdfs = ["runbook.pdf", "network_guide.pdf", "error_codes.pdf"]
    chunks = ingest_multiple_pdfs(pdfs)

    vs = VectorStore()
    vs.build_index(chunks)
    vs.save("it_knowledge_base")

    # Test search
    results = vs.search("server keeps restarting unexpectedly", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n─── Result {i} ───\n{r[:300]}")
