IT Support AI Agent
RAG-Powered Document Ingestion & Intelligent Troubleshooting
Full Implementation Guide
Architecture Overview
The RAG pipeline connects four stages: PDF ingestion, embedding, vector storage, and AI generation. Each stage feeds the next, giving Claude grounded, document-backed answers.
Code
💡 Key Idea: Claude only sees the most relevant document chunks per query — not the entire PDF. This keeps responses fast, accurate, and directly grounded in your documentation.
File 1 — ingest_pdf.py
Purpose
Extracts raw text from PDF files (preserving page numbers) and splits it into overlapping chunks suitable for embedding. This is the entry point for all document knowledge.
Install
Code
Key Parameters
chunk_size=500 — tokens per chunk; tune per document type
chunk_overlap=50 — preserves context at chunk boundaries (10–15% of chunk_size)
separators=["\n\n", "\n", ".", " "] — splits on paragraphs first, then sentences
📐 Chunk Size Guide: Error codes / API refs: 200–300 tokens | Runbooks: 500–700 | Architecture docs: 800–1000
Usage
Python
File 2 — vector_store.py
Purpose
Encodes text chunks as 384-dimensional vectors using a SentenceTransformer model, stores them in a FAISS index for millisecond similarity search, and persists the index to disk so it only needs to be built once.
Install
Code
Key Methods
build_index(chunks) — embeds all chunks and adds to FAISS IndexFlatIP
search(query, top_k=5) — returns the top-k most relevant chunks
save(path) / load(path) — persist and reload the index between sessions
Usage
Python
File 3 — it_agent.py
Purpose
The core AI agent. Retrieves relevant documentation chunks for each user issue, injects them into Claude's system prompt as grounded context, and maintains a multi-turn conversation history so follow-up questions are supported.
Install
Code
System Prompt Design
Context is injected as numbered excerpts so Claude can reference them
Claude is instructed to cite excerpt numbers in its response
Explicit instruction: do NOT fabricate information outside the context
Always ends with a Verification section to confirm the fix worked
Usage
Python
File 4 — main.py
Purpose
The CLI orchestrator. Accepts --build to ingest PDFs and create the index, or starts the agent directly with an existing index. Supports an interactive REPL with reset and exit commands.
CLI Usage
Bash
Session Commands
reset — clears conversation history to start a fresh problem
exit / quit — ends the session
Production Enhancements
For production deployments, consider these upgrades across six key areas:
Area
Recommendation
Vector DB
Replace FAISS with Pinecone, Weaviate, or pgvector for persistence & scale
Embeddings
Use text-embedding-3-large (OpenAI) or Cohere for higher accuracy
Chunking
Add metadata (filename, page number) to each chunk for source citations
Reranking
Add a cross-encoder reranker after retrieval to boost precision
Hybrid Search
Combine BM25 (keyword) + vector search for better recall
Guardrails
Add hallucination checks — verify every claim is in retrieved context
Quick Start Checklist
Install all dependencies: pypdf, langchain, sentence-transformers, faiss-cpu, anthropic
Set ANTHROPIC_API_KEY environment variable
Place your IT PDF documents in the project folder
Run: python main.py --build --pdfs your_docs.pdf
Describe an IT issue and receive a grounded, step-by-step solution
📁 File Summary: ingest_pdf.py — PDF parsing & chunking | vector_store.py — embeddings & FAISS | it_agent.py — Claude RAG agent | main.py — CLI orchestrator# AI-RAG-angentic-IT