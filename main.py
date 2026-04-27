"""
main.py — Step 4: Full Pipeline Orchestrator
=============================================
Ties together ingestion, indexing, and the agent.
Run this file to build the knowledge base and start
the interactive IT support session.

Usage:
    # Build index from PDFs (first run)
    python main.py --build --pdfs runbook.pdf network_guide.pdf error_codes.pdf

    # Start agent with existing index
    python main.py

    # Rebuild index and start agent
    python main.py --build --pdfs runbook.pdf

Install all dependencies:
    pip install pypdf langchain sentence-transformers faiss-cpu numpy anthropic \
        --break-system-packages
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

import argparse
import os
import sys

from ingest_pdf import ingest_multiple_pdfs
from vector_store import VectorStore
from it_agent import ITSupportAgent

# ── Constants ────────────────────────────────────────────────────
INDEX_PATH = "it_knowledge_base"   # base filename for FAISS index + chunks
DEFAULT_PDFS = [                   # fallback PDF list if none passed via CLI
    "runbook.pdf",
    "network_guide.pdf",
    "error_codes.pdf",
]


# ── Build phase ──────────────────────────────────────────────────

def build_knowledge_base(pdf_paths: list[str]) -> VectorStore:
    """
    Ingest PDFs, embed chunks, build FAISS index, and save to disk.

    Args:
        pdf_paths: List of paths to PDF documents.

    Returns:
        Populated VectorStore instance.
    """
    print("\n" + "═" * 60)
    print("  PHASE 1 — PDF INGESTION")
    print("═" * 60)
    chunks = ingest_multiple_pdfs(pdf_paths)

    if not chunks:
        print("❌  No chunks produced. Check your PDF paths and content.")
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  PHASE 2 — EMBEDDING & INDEXING")
    print("═" * 60)
    vs = VectorStore()
    vs.build_index(chunks)
    vs.save(INDEX_PATH)

    return vs


# ── Load phase ───────────────────────────────────────────────────

def load_knowledge_base() -> VectorStore:
    """
    Load an existing FAISS index from disk.

    Returns:
        Populated VectorStore instance.
    """
    index_file = f"{INDEX_PATH}.index"
    if not os.path.exists(index_file):
        print(f"❌  No index found at '{index_file}'.")
        print("    Run with --build to create it first.")
        sys.exit(1)

    vs = VectorStore()
    vs.load(INDEX_PATH)
    return vs


# ── Interactive agent loop ───────────────────────────────────────

def run_agent(vs: VectorStore) -> None:
    """
    Start the interactive IT support session.

    Args:
        vs: Loaded VectorStore to use for retrieval.
    """
    agent = ITSupportAgent(vs, top_k=5)

    print("\n" + "═" * 60)
    print("  IT SUPPORT AGENT — READY")
    print("═" * 60)
    print("Commands:  'exit' | 'quit' → end session")
    print("           'reset'         → clear conversation history")
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

        print("\n🤖  Agent:\n")
        try:
            solution = agent.solve(issue)
            print(solution)
        except Exception as e:
            print(f"❌  Error from agent: {e}")

        print("\n" + "─" * 60 + "\n")


# ── Entry point ──────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IT Support RAG Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Ingest PDFs and rebuild the FAISS index before starting the agent.",
    )
    parser.add_argument(
        "--pdfs",
        nargs="+",
        metavar="PDF",
        default=DEFAULT_PDFS,
        help="PDF file paths to ingest (only used with --build).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.build:
        vs = build_knowledge_base(args.pdfs)
    else:
        vs = load_knowledge_base()

    run_agent(vs)


if __name__ == "__main__":
    main()
