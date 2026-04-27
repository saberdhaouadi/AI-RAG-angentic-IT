"""
vector_store.py — Step 2: Embedding & Vector Store (with Incremental Ingestion)
===============================================================================
Embeds text chunks using a sentence-transformer model and stores them
in a FAISS index for fast similarity search.

Supports:
  - build_index(chunks)      → build from scratch
  - add_chunks(chunks)       → append to existing index (incremental)
  - remove_source(name)      → remove a PDF's chunks and rebuild
  - list_sources()           → show all ingested PDF filenames
  - search(query, top_k)     → similarity search
  - save(path)               → persist index + chunks to disk
  - load(path)               → reload from disk

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
        model:    Loaded SentenceTransformer instance.
        index:    FAISS index for fast nearest-neighbour search.
        chunks:   List of original text chunks (parallel to index).
        sources:  List of source labels per chunk (e.g. PDF filename).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"🔄  Loading embedding model: {model_name} …")
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.chunks: list[str] = []
        self.sources: list[str] = []

    # ── Internal embed helper ─────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts and return normalised float32 embeddings."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    # ── Build from scratch ────────────────────────────────────────

    def build_index(self, chunks: list[str], source: str = "unknown") -> None:
        """
        Encode chunks and build a brand-new FAISS index.
        Wipes any existing index.

        Args:
            chunks: List of text strings to embed and store.
            source: Label for these chunks (e.g. PDF filename).
        """
        if not chunks:
            raise ValueError("chunks list is empty — nothing to index.")

        self.chunks = chunks
        self.sources = [source] * len(chunks)

        print(f"⚙️   Embedding {len(chunks)} chunks …")
        embeddings = self._embed(chunks)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"✅  Index built: {self.index.ntotal} vectors, dim={dimension}")

    # ── Incremental add ───────────────────────────────────────────

    def add_chunks(self, chunks: list[str], source: str = "unknown") -> None:
        """
        Append new chunks to an existing index without rebuilding.
        If no index exists yet, builds one automatically.

        Args:
            chunks: New text chunks to add.
            source: Label for these chunks (e.g. PDF filename).
        """
        if not chunks:
            print("⚠️   No chunks to add.")
            return

        if source != "unknown" and source in self.sources:
            print(f"⚠️   '{source}' is already in the index. Skipping.")
            print(f"     Use remove_source('{source}') first to re-ingest it.")
            return

        print(f"➕  Adding {len(chunks)} chunks from '{source}' …")
        embeddings = self._embed(chunks)

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            print(f"    Created new index (dim={dimension})")

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.sources.extend([source] * len(chunks))

        print(f"✅  Index now has {self.index.ntotal} vectors total.")

    # ── Remove a source ───────────────────────────────────────────

    def remove_source(self, source: str) -> None:
        """
        Remove all chunks belonging to a specific source and rebuild index.

        Args:
            source: The source label to remove (e.g. PDF filename).
        """
        if source not in self.sources:
            print(f"⚠️   Source '{source}' not found in index.")
            print(f"     Available: {self.list_sources()}")
            return

        before = len(self.chunks)
        paired = [(c, s) for c, s in zip(self.chunks, self.sources) if s != source]

        if not paired:
            print(f"⚠️   Removing '{source}' would empty the index.")
            return

        self.chunks, self.sources = map(list, zip(*paired))
        removed = before - len(self.chunks)

        print(f"🗑️   Removed {removed} chunks from '{source}'. Rebuilding index …")
        embeddings = self._embed(self.chunks)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print(f"✅  Index rebuilt: {self.index.ntotal} vectors remaining.")

    # ── List sources ──────────────────────────────────────────────

    def list_sources(self) -> list[str]:
        """Return a deduplicated list of all ingested source labels."""
        seen = []
        for s in self.sources:
            if s not in seen:
                seen.append(s)
        return seen

    # ── Search ────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Return the top-k most relevant chunks for a query."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() or load() first.")

        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec, dtype="float32")
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)

        return [self.chunks[idx] for idx in indices[0] if idx != -1]

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str = "vector_store") -> None:
        """Persist FAISS index, chunks, and sources to disk."""
        if self.index is None:
            raise RuntimeError("No index to save.")

        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "sources": self.sources}, f)

        print(f"💾  Saved index to {path}.index + {path}.pkl")
        print(f"    Sources: {self.list_sources()}")

    def load(self, path: str = "vector_store") -> None:
        """Load a previously saved index from disk."""
        index_file = f"{path}.index"
        chunks_file = f"{path}.pkl"

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

        self.index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            data = pickle.load(f)

        # Support old format (plain list) and new format (dict with sources)
        if isinstance(data, dict):
            self.chunks = data["chunks"]
            self.sources = data["sources"]
        else:
            self.chunks = data
            self.sources = ["unknown"] * len(data)

        print(f"📂  Loaded index: {self.index.ntotal} vectors")
        print(f"    Sources: {self.list_sources()}")
