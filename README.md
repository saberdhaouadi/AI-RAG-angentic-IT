# AI-RAG-Agentic-IT

> **RAG-Powered Document Ingestion & Intelligent Troubleshooting for IT Support**

An enterprise-grade AI agent that uses Retrieval-Augmented Generation (RAG) to provide grounded, document-backed IT support solutions. Powered by local LLMs (Gemma3/Gemma4), FAISS vector search, and multi-turn conversation memory — runs entirely on your machine.

---

## Table of Contents

- [Why RAG for IT Support?](#why-rag-for-it-support)
- [Architecture Overview](#architecture-overview)
- [Components](#components)
- [Installation](#installation)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Production Enhancements](#production-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Why RAG for IT Support?

Traditional chatbots either:
- **Hallucinate** answers not grounded in your documentation
- **Miss context** from your specific runbooks, architecture docs, and error catalogs
- **Outdated** — require manual retraining on documentation changes

This RAG solution:
✅ **Grounds every answer** in your actual IT documentation  
✅ **Cites sources** — local LLM references specific excerpts  
✅ **Stays current** — re-ingest PDFs as documentation updates  
✅ **Fast retrieval** — millisecond vector similarity search  
✅ **Privacy-first** — runs entirely on local LLMs (Gemma3/Gemma4), no external API calls

---

## Architecture Overview

The RAG pipeline connects four stages: PDF ingestion, embedding, vector storage, and AI generation. Each stage feeds the next, giving your local LLM grounded, document-backed answers.

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────────┐
│ PDF Files   │────▶│ Chunk & Parse│────▶│  Embeddings  │────▶│ Vector Store    │
│             │     │ (ingest_pdf) │     │(SentenceXForm)     │   (FAISS)       │
└─────────────┘     └──────────────┘     └──────────────┘     └─────────────────┘
                                                                        ▲
                                                                        │
                                      ┌──────────────────────────────────┤
                                      │                                  │
                           ┌──────────▼──────────┐           ┌──────────┴──────────────┐
                           │   User Query        │           │  it_agent.py            │
                           │  (IT Issue)         │────────┬──▶│  (Local LLM RAG Agent)  │
                           └─────────────────────┘        │   │  (Gemma3/Gemma4)       │
                                                          │   └─────────────────────────┘
                                                          │            │
                                                          │            ▼
                                                          │   ┌──────────────────┐
                                                          │   │ Grounded Answer  │
                                                          └──▶│ + Citations      │
                                                              └──────────────────┘
```

💡 **Key Idea**: The local LLM only sees the most relevant document chunks per query — not the entire PDF. This keeps responses fast, accurate, and directly grounded in your documentation. No external API calls, no data leakage.

---

## Components

### File 1 — `ingest_pdf.py`

**Purpose**  
Extracts raw text from PDF files (preserving page numbers) and splits it into overlapping chunks suitable for embedding. This is the entry point for all document knowledge.

**Key Parameters**
- `chunk_size=500` — tokens per chunk; tune per document type
- `chunk_overlap=50` — preserves context at chunk boundaries (10–15% of chunk_size)
- `separators=["\n\n", "\n", ".", " "]` — splits on paragraphs first, then sentences

📐 **Chunk Size Guide:**
| Document Type | Recommended Size |
|---|---|
| Error codes / API refs | 200–300 tokens |
| Runbooks | 500–700 tokens |
| Architecture docs | 800–1000 tokens |

**Example Usage**
```python
from ingest_pdf import load_and_chunk_pdfs

# Load and chunk your IT documentation
chunks = load_and_chunk_pdfs(
    pdf_path="docs/IT_Runbook.pdf",
    chunk_size=500,
    chunk_overlap=50
)
print(f"Created {len(chunks)} chunks")
# Output: Created 487 chunks
```

---

### File 2 — `vector_store.py`

**Purpose**  
Encodes text chunks as 384-dimensional vectors using a SentenceTransformer model, stores them in a FAISS index for millisecond similarity search, and persists the index to disk for reuse.

**Key Methods**
- `build_index(chunks)` — embeds all chunks and adds to FAISS IndexFlatIP
- `search(query, top_k=5)` — returns the top-k most relevant chunks
- `save(path)` / `load(path)` — persist and reload the index between sessions

**Example Usage**
```python
from vector_store import VectorStore

# Initialize and build the vector store
vs = VectorStore(model_name="all-MiniLM-L6-v2")
vs.build_index(chunks)
vs.save("faiss_index")

# Search for relevant documentation
query = "How do I troubleshoot high CPU usage?"
results = vs.search(query, top_k=5)
for i, chunk in enumerate(results, 1):
    print(f"{i}. {chunk[:100]}...")
```

---

### File 3 — `it_agent.py`

**Purpose**  
The core AI agent. Retrieves relevant documentation chunks for each user issue, injects them into the local LLM's system prompt as grounded context, and maintains multi-turn conversation history for coherent troubleshooting threads.

**System Prompt Design**
- Context is injected as numbered excerpts so the LLM can reference them
- LLM is instructed to cite excerpt numbers in its response
- Explicit instruction: do NOT fabricate information outside the context
- Always ends with a Verification section to confirm the fix worked

**Example Usage**
```python
from it_agent import ITAgent

agent = ITAgent(
    vector_store=vs,
    model="gemma:7b",  # Local Gemma model via Ollama
    base_url="http://localhost:11434"  # Ollama server
)

# Multi-turn conversation
response = agent.chat("My database is running slow. What should I check?")
print(response)
# Output:
# Based on excerpt [2] and [5], here are the key troubleshooting steps:
# 1. Check query execution times...
# 2. Review index fragmentation...
# [Verification: Apply these checks and report back with results]
```

---

### File 4 — `main.py`

**Purpose**  
The CLI orchestrator. Accepts `--build` to ingest PDFs and create the index, or starts the agent directly with an existing index. Supports interactive REPL with reset and exit commands.

**CLI Usage**
```bash
# Build vector index from PDFs (run once or when docs update)
python main.py --build --pdfs docs/Runbook.pdf docs/Architecture.pdf

# Start interactive agent session
python main.py

# In the REPL:
> My disk is at 95% capacity. What should I do?
[Agent responds with grounded solution]

> reset                    # Clear conversation history
> exit                     # End session
```

**Session Commands**
- `reset` — clears conversation history to start a fresh problem
- `exit` / `quit` — ends the session

---

## Installation

### Prerequisites
- Python 3.9+
- Local LLM server running Gemma3/Gemma4 (via Ollama, LM Studio, or similar)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/saberdhaouadi/AI-RAG-angentic-IT.git
   cd AI-RAG-angentic-IT
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up local LLM (Ollama example)**
   ```bash
   # Install Ollama from https://ollama.ai
   # Pull Gemma model
   ollama pull gemma:7b
   
   # Start Ollama server (runs on http://localhost:11434 by default)
   ollama serve
   ```

### Dependencies

The project requires:
- `pypdf` — PDF parsing
- `langchain` — LLM orchestration
- `sentence-transformers` — embedding model
- `faiss-cpu` — vector similarity search
- `ollama` or `openai` (for local LLM compatibility)

See `requirements.txt` for pinned versions.

---

## System Requirements

Here are the full system requirements to run this stack locally:

### Minimum Requirements (gemma3 4B — default)

| Component | Minimum |
|---|---|
| RAM | 8 GB |
| Disk | 10 GB free (5GB model + index + OS) |
| CPU | 4-core x86_64 (Intel/AMD) |
| GPU | None required — CPU-only works |
| OS | Ubuntu 20.04+, macOS 12+, Windows 10/11 (WSL2) |
| Python | 3.10+ |

⚠️ **CPU-only inference on 4B is slow** — expect 15–60 seconds per response depending on hardware.

### Recommended (for decent speed)

| Component | Recommended |
|---|---|
| RAM | 16 GB |
| GPU | NVIDIA GPU with 8 GB VRAM (e.g. RTX 3060/4060) |
| Disk | 20 GB free |
| CPU | 8-core modern CPU |

✅ **With an 8GB VRAM GPU, responses drop to 2–5 seconds.**

### By Model Variant

| Model | RAM (CPU) | VRAM (GPU) | Speed |
|---|---|---|---|
| gemma3 (4B) | 8 GB | 4–6 GB | Fastest |
| gemma3:12b | 16 GB | 10–12 GB | Balanced |
| gemma3:27b | 32 GB | 20–24 GB | Best quality |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start your local LLM server (Ollama)
# In a separate terminal:
ollama serve

# 3. Pull Gemma model if not already done
ollama pull gemma:7b

# 4. Place IT documentation PDFs in the project folder
cp /path/to/your/docs/*.pdf ./docs/

# 5. Build the vector index (run once)
python main.py --build --pdfs docs/*.pdf

# 6. Start the agent
python main.py

# 7. Describe an IT issue and get a grounded solution
> Our production database is unresponsive. What's the diagnostic process?
```

---

## Configuration

### Environment Variables

```bash
LOCAL_LLM_BASE_URL     # Optional: Base URL for local LLM (default: http://localhost:11434)
LOCAL_LLM_MODEL        # Optional: Model name (default: gemma:7b)
FAISS_INDEX_PATH       # Optional: Path to vector store (default: ./faiss_index)
CHUNK_SIZE             # Optional: Tokens per chunk (default: 500)
CHUNK_OVERLAP          # Optional: Chunk overlap in tokens (default: 50)
```

### Tuning Parameters

Edit these in the respective files:

**`ingest_pdf.py`**
```python
chunk_size = 500         # Increase for longer context, decrease for precision
chunk_overlap = 50       # 10-15% of chunk_size is typical
```

**`vector_store.py`**
```python
model_name = "all-MiniLM-L6-v2"  # Faster but lower accuracy
# or use "all-mpnet-base-v2" for higher accuracy (slower)
```

**`it_agent.py`**
```python
model = "gemma:7b"               # Local model name
base_url = "http://localhost:11434"  # Ollama server URL
top_k = 5                        # Number of retrieved chunks (increase for recall, decrease for brevity)
temperature = 0.3                # Lower = more deterministic, higher = more creative
```

---

## Usage Examples

### Example 1: Database Troubleshooting

```
User: Our database is timing out on queries. What should we check first?

Agent (grounded response):
According to excerpt [3] from our Troubleshooting Runbook, the first diagnostic steps are:

1. Check Query Execution Plans (excerpt [3])
   - Run: EXPLAIN ANALYZE on slow queries
   - Look for sequential scans on large tables
   
2. Verify Index Health (excerpt [5])
   - Execute: REINDEX CONCURRENTLY
   - Monitor: index fragmentation < 10%

3. Review Connection Pool (excerpt [7])
   - Check: active connections vs max_connections setting
   - If stuck: kill idle sessions

[Verification: Run these checks and report connection pool status]
```

### Example 2: Multi-turn Conversation

```
User: I found 50 idle connections. What's next?

Agent:
Based on the idle connections you found, excerpt [9] recommends:

1. Identify the source (are they from a specific application?)
2. Either: Kill the idle connections or increase pool settings

Would you like me to explain how to safely terminate idle connections?

User: Yes, how do I safely kill them?

Agent:
Follow excerpt [11]:
```sql
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'idle' 
AND query_start < NOW() - INTERVAL '30 minutes';
```

[Verification: Run this and check if query performance improves]
```

---

## Troubleshooting

### Issue: "Connection refused to localhost:11434"
**Solution:**
```bash
# Ensure Ollama is running:
ollama serve

# Or start it in a separate terminal and verify:
curl http://localhost:11434/api/tags
```

### Issue: "Model 'gemma:7b' not found"
**Solution:**
```bash
# Pull the model first
ollama pull gemma:7b

# List available models:
ollama list
```

### Issue: "No such file or directory: faiss_index"
**Solution:**
```bash
# Build the index first
python main.py --build --pdfs docs/*.pdf
```

### Issue: "PDF parsing error - unsupported encoding"
**Solution:**
- Ensure PDFs are text-based (not scanned images)
- Try re-saving PDF in UTF-8 encoding
- Use OCR tool (e.g., Tesseract) for scanned docs first

### Issue: "Vector search returns irrelevant results"
**Solutions:**
- Increase `chunk_size` if documents have important context spread across pages
- Reduce `top_k` to 3–4 to return only the most relevant chunks
- Use the production "Reranking" enhancement (see below)

### Issue: "Slow embedding or search"
**Solutions:**
- Use `faiss-cpu` for development; switch to GPU version for scale:
   ```bash
   pip install faiss-gpu
   ```
- Switch to faster (but lower-quality) embeddings: `all-MiniLM-L6-v2`
- Try a smaller Gemma model: `gemma:2b` instead of `gemma:7b`

---

## Production Enhancements

For production deployments, consider these upgrades across six key areas:

| Area | Recommendation | Why | Trade-offs |
|---|---|---|---|
| **Vector DB** | Replace FAISS with Pinecone, Weaviate, or pgvector | Persistence, scalability, multi-user indexing | Managed services cost money; added latency |
| **Embeddings** | Use `text-embedding-3-large` (OpenAI) or Cohere | Higher accuracy (3584 dims vs 384) | API calls per embed add latency & cost |
| **Chunking** | Add metadata (filename, page number) to chunks | Enable precise source citations | Requires storing metadata in vector DB |
| **Reranking** | Add cross-encoder reranker after retrieval | Boost precision on marginal relevance | Adds ~100ms latency per query |
| **Hybrid Search** | Combine BM25 (keyword) + vector search | Better recall on exact terms | Complexity; two search paths to optimize |
| **Guardrails** | Add hallucination checks — verify claims in context | Prevent false information | Requires additional LLM call (~0.1–0.2s) |

### Example: Production Upgrade Path
1. **Month 1:** Add metadata to chunks (quick win)
2. **Month 2:** Implement cross-encoder reranking for precision
3. **Month 3:** Migrate to Pinecone for scale
4. **Month 4:** Optimize embedding model and local LLM for inference speed

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes with clear messages
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Areas for Contribution
- [ ] Add support for DOCX, Markdown, and plain text documents
- [ ] Implement caching layer for frequent queries
- [ ] Add logging and monitoring
- [ ] Create Docker containerization
- [ ] Build web UI for agent
- [ ] Add eval framework for answer quality
- [ ] Support for other local LLMs (Llama, Mistral, etc.)

---

## File Summary

| File | Purpose |
|---|---|
| `ingest_pdf.py` | PDF parsing & chunking with overlap |
| `vector_store.py` | SentenceTransformer embeddings & FAISS indexing |
| `it_agent.py` | Local LLM RAG agent with multi-turn memory |
| `main.py` | CLI orchestrator & interactive REPL |
| `requirements.txt` | Python dependencies |

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

## Support

- **Issues:** Open a GitHub issue for bugs or feature requests
- **Docs:** See Architecture Overview for design details
- **Questions:** Check Troubleshooting section first
- **Local LLM Setup:** See [Ollama Documentation](https://ollama.ai)

---

**Built with ❤️ for IT teams who want grounded, accurate, privacy-respecting support automation with local LLMs.**
