"""
main.py — Full Pipeline Orchestrator (with Incremental Ingestion)
=================================================================

Usage:
    # First run — build index from PDFs
    python main.py --build --pdfs doc1.pdf doc2.pdf

    # Add new PDFs to existing index WITHOUT rebuilding
    python main.py --add --pdfs new_doc.pdf another_doc.pdf

    # List all ingested sources
    python main.py --list

    # Remove a source from the index
    python main.py --remove doc1.pdf

    # Start agent with existing index
    python main.py

    # Use a specific Gemma variant
    python main.py --model gemma3:27b

Install all dependencies:
    pip install pypdf langchain-text-splitters sentence-transformers faiss-cpu numpy ollama \
        --break-system-packages
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull gemma3
"""

import argparse
import os
import sys

from ingest import ingest_pdf, ingest_multiple_pdfs
from vector_store import VectorStore
from it_agent import ITSupportAgent

INDEX_PATH = "it_knowledge_base"
DEFAULT_PDFS = ["runbook.pdf", "network_guide.pdf", "error_codes.pdf"]


# ── Build (from scratch) ─────────────────────────────────────────

def build_knowledge_base(pdf_paths: list[str]) -> VectorStore:
    """Ingest all PDFs and build a brand-new FAISS index."""
    print("\n" + "═" * 60)
    print("  PHASE 1 — PDF INGESTION (full rebuild)")
    print("═" * 60)

    vs = VectorStore()
    for path in pdf_paths:
        try:
            chunks = ingest_pdf(path)
            vs.add_chunks(chunks, source=os.path.basename(path))
        except (FileNotFoundError, ValueError) as e:
            print(f"⚠️   Skipping {path}: {e}")

    if not vs.chunks:
        print("❌  No chunks produced. Check your PDF paths.")
        sys.exit(1)

    vs.save(INDEX_PATH)
    return vs


# ── Add (incremental) ────────────────────────────────────────────

def add_to_knowledge_base(pdf_paths: list[str]) -> VectorStore:
    """Load existing index and append new PDFs without rebuilding."""
    print("\n" + "═" * 60)
    print("  INCREMENTAL INGESTION — adding to existing index")
    print("═" * 60)

    vs = load_knowledge_base()

    for path in pdf_paths:
        try:
            chunks = ingest_pdf(path)
            vs.add_chunks(chunks, source=os.path.basename(path))
        except (FileNotFoundError, ValueError) as e:
            print(f"⚠️   Skipping {path}: {e}")

    vs.save(INDEX_PATH)
    return vs


# ── Load ─────────────────────────────────────────────────────────

def load_knowledge_base() -> VectorStore:
    """Load an existing FAISS index from disk."""
    index_file = f"{INDEX_PATH}.index"
    if not os.path.exists(index_file):
        print(f"❌  No index found at '{index_file}'.")
        print("    Run with --build to create it first.")
        sys.exit(1)

    vs = VectorStore()
    vs.load(INDEX_PATH)
    return vs


# ── Remove source ────────────────────────────────────────────────

def remove_source(source: str) -> VectorStore:
    """Remove a PDF source from the index and save."""
    vs = load_knowledge_base()
    vs.remove_source(source)
    vs.save(INDEX_PATH)
    return vs


# ── Interactive agent loop ───────────────────────────────────────

def run_agent(vs: VectorStore, model: str) -> None:
    """Start the interactive IT support session."""
    agent = ITSupportAgent(vs, top_k=5, model=model)

    print("\n" + "═" * 60)
    print(f"  IT SUPPORT AGENT — READY  [{agent.model}]")
    print("═" * 60)
    print("Commands:  'exit' | 'quit' → end session")
    print("           'reset'         → clear conversation history")
    print("           'sources'       → list ingested documents")
    print("─" * 60 + "\n")

    while True:
        try:
            issue = input("🖥️  Describe your IT issue:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Session ended.")
            break

        if not issue:
            continue
        if issue.lower() in ("exit", "quit"):
            print("👋  Goodbye!")
            break
        if issue.lower() == "reset":
            agent.reset_conversation()
            continue
        if issue.lower() == "sources":
            print("\n📄  Ingested sources:")
            for s in vs.list_sources():
                print(f"    • {s}")
            print()
            continue

        print("\n🤖  Agent:\n")
        try:
            solution = agent.solve(issue)
            print(solution)
        except Exception as e:
            print(f"❌  Error from agent: {e}")

        print("\n" + "─" * 60 + "\n")


# ── CLI ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IT Support RAG Agent — powered by local Gemma (Ollama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Ingest PDFs and rebuild the index from scratch.")
    parser.add_argument(
        "--add", action="store_true",
        help="Add new PDFs to the existing index (no rebuild).")
    parser.add_argument(
        "--list", action="store_true",
        help="List all ingested sources and exit.")
    parser.add_argument(
        "--remove", metavar="SOURCE",
        help="Remove a source (PDF filename) from the index, then start agent.")
    parser.add_argument(
        "--pdfs", nargs="+", metavar="PDF", default=DEFAULT_PDFS,
        help="PDF file paths to ingest (used with --build or --add).")
    parser.add_argument(
        "--model", default=os.getenv("OLLAMA_MODEL", "gemma3"), metavar="MODEL",
        help="Ollama model tag. Examples: gemma3, gemma3:12b, gemma3:27b")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --list: just print sources and exit
    if args.list:
        vs = load_knowledge_base()
        print("\n📄  Ingested sources:")
        for s in vs.list_sources():
            print(f"    • {s}")
        return

    # --remove: drop a source, save, then start agent
    if args.remove:
        vs = remove_source(args.remove)
        run_agent(vs, model=args.model)
        return

    # --build: full rebuild
    if args.build:
        vs = build_knowledge_base(args.pdfs)

    # --add: incremental
    elif args.add:
        vs = add_to_knowledge_base(args.pdfs)

    # default: load existing index
    else:
        vs = load_knowledge_base()

    run_agent(vs, model=args.model)


if __name__ == "__main__":
    main()
