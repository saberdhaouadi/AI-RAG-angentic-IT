"""
ingest_pdf.py — Step 1: PDF Ingestion & Chunking
================================================
Extracts text from PDF files and splits them into
overlapping chunks suitable for embedding.

Install:
    pip install pypdf langchain --break-system-packages
"""

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def ingest_pdf(pdf_path: str) -> list[str]:
    """
    Extract text from a PDF and split into chunks.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of text chunks ready for embedding.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ── 1. Extract raw text ──────────────────────────────────────
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            raw_text += f"\n[Page {page_number}]\n{page_text}\n"

    if not raw_text.strip():
        raise ValueError(f"No extractable text found in: {pdf_path}")

    # ── 2. Chunk the text ────────────────────────────────────────
    # Chunk size guidelines for IT documents:
    #   - Error codes / API refs  → 200–300 tokens
    #   - Runbooks / procedures   → 500–700 tokens
    #   - Architecture docs       → 800–1000 tokens
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,         # tokens per chunk
        chunk_overlap=50,       # overlap to preserve context at boundaries
        separators=["\n\n", "\n", ".", " "]  # split order: paragraphs first
    )
    chunks = splitter.split_text(raw_text)

    print(f"✅  {os.path.basename(pdf_path)}: {len(reader.pages)} pages → {len(chunks)} chunks")
    return chunks


def ingest_multiple_pdfs(pdf_paths: list[str]) -> list[str]:
    """
    Ingest multiple PDFs and return all chunks combined.

    Args:
        pdf_paths: List of PDF file paths.

    Returns:
        Combined list of text chunks from all documents.
    """
    all_chunks = []
    for path in pdf_paths:
        try:
            chunks = ingest_pdf(path)
            all_chunks.extend(chunks)
        except (FileNotFoundError, ValueError) as e:
            print(f"⚠️  Skipping {path}: {e}")

    print(f"\n📄  Total chunks across all documents: {len(all_chunks)}")
    return all_chunks


# ── Example usage ────────────────────────────────────────────────
if __name__ == "__main__":
    # Replace with your actual PDF paths
    pdfs = [
        "runbook.pdf",
        "network_guide.pdf",
        "error_codes.pdf",
    ]

    chunks = ingest_multiple_pdfs(pdfs)

    # Preview first 3 chunks
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n─── Chunk {i} ───\n{chunk[:300]}...")
