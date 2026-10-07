"""
Phase A — Ingestion pipeline

Run:
    python ingest.py path/to/your/file.pdf

What this script does, in order:
  1. Read the PDF with PyMuPDF (page by page)
  2. Clean the text on each page
  3. Chunk the text with LangChain
  4. Embed each chunk with Google Generative AI
  5. Upsert vectors into Pinecone
"""

import re
import sys
import uuid

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

import config
from embedding_provider import get_embedder


# ── Step 1: Extract text from PDF ────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[dict]:
    """
    Open the PDF and return a list of pages.
    Each page is a dict:  { "page": int, "text": str }
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        raw_text = page.get_text()          # extract raw text from this page
        pages.append({
            "page": i + 1,                  # 1-based page number
            "text": raw_text,
        })
    doc.close()
    return pages


# ── Step 2: Clean text ────────────────────────────────────────────────────────

def clean(text: str) -> str:
    """
    Collapse all extra whitespace (newlines, tabs, double spaces) into
    a single space so LangChain chunking works cleanly.
    """
    return re.sub(r"\s+", " ", text).strip()


# ── Step 3: Chunk ─────────────────────────────────────────────────────────────

def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split each page's text into smaller chunks.
    We chunk per page so we can keep track of the page number.
    Returns a list of chunks:
        { "chunk_id": str, "source": str, "page": int, "text": str }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks = []
    for page in pages:
        cleaned = clean(page["text"])
        if not cleaned:
            continue                        # skip completely empty pages

        # LangChain splits one page of text into a list of Document objects
        docs = splitter.create_documents([cleaned])

        for doc in docs:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),   # unique ID for Pinecone
                "page":     page["page"],
                "text":     doc.page_content,
            })

    return chunks


# ── Step 4 + 5: Embed and upsert into Pinecone ───────────────────────────────

def embed_and_upsert(chunks: list[dict], source_name: str) -> int:
    """
    - Embed every chunk with Google Generative AI Embeddings
    - Upsert (insert or update) the vectors into Pinecone

    source_name: the filename/path — stored as metadata so you know
                 which document each chunk came from.
    """

    # 4a. Connect to Google Embeddings
    embedder = get_embedder()

    # 4b. Connect to Pinecone
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    # 5. Embed and upsert in small batches (Pinecone recommends ≤ 100 per call)
    BATCH_SIZE = 50
    total = len(chunks)
    index = None

    upserted_total = 0
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]

        # Get the raw text from each chunk in this batch
        texts = [c["text"] for c in batch]

        # Embed all texts in one API call
        vectors = embedder.embed_documents(texts)
        vector_dim = len(vectors[0]) if vectors else 0

        # Create/check index only after we know actual embedding dimension
        if index is None:
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if config.PINECONE_INDEX not in existing_indexes:
                print(f"Index '{config.PINECONE_INDEX}' not found — creating it...")
                pc.create_index(
                    name=config.PINECONE_INDEX,
                    dimension=vector_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                print(f"Index created with dimension {vector_dim}.")
            else:
                desc = pc.describe_index(config.PINECONE_INDEX)
                existing_dim = getattr(desc, "dimension", None)
                if existing_dim != vector_dim:
                    if not config.ALLOW_INDEX_RECREATE:
                        raise ValueError(
                            f"Index '{config.PINECONE_INDEX}' dimension is {existing_dim}, "
                            f"but embedding model returns {vector_dim}. "
                            "Set ALLOW_INDEX_RECREATE=true or change index/model."
                        )
                    print(
                        f"Index dimension mismatch ({existing_dim} vs {vector_dim}). "
                        "Recreating index..."
                    )
                    pc.delete_index(config.PINECONE_INDEX)
                    pc.create_index(
                        name=config.PINECONE_INDEX,
                        dimension=vector_dim,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                    )
                    print(f"Index recreated with dimension {vector_dim}.")

            index = pc.Index(config.PINECONE_INDEX)

        # Build the list Pinecone expects: (id, vector, metadata)
        upsert_data = []
        for chunk, vector in zip(batch, vectors):
            upsert_data.append({
                "id":     chunk["chunk_id"],
                "values": vector,
                "metadata": {
                    "source": source_name,
                    "page":   chunk["page"],
                    "text":   chunk["text"],   # store text so we can retrieve it later
                },
            })

        index.upsert(vectors=upsert_data)
        upserted_total += len(upsert_data)
        print(f"  Upserted chunks {start + 1}–{start + len(batch)} / {total}")

    return upserted_total


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest(pdf_path: str) -> dict:
    print(f"\n=== Ingesting: {pdf_path} ===")

    print("Step 1: Extracting pages from PDF...")
    pages = extract_pages(pdf_path)
    print(f"  Found {len(pages)} pages.")

    print("Step 2 & 3: Cleaning and chunking...")
    chunks = chunk_pages(pages)
    print(f"  Produced {len(chunks)} chunks.")

    if not chunks:
        print(
            "No chunks were produced, so nothing will be written to Pinecone.\n"
            "This usually means the PDF has little/no extractable text.\n"
            "Try another PDF or lower CHUNK_SIZE in .env."
        )
        return {
            "source": pdf_path,
            "page_count": len(pages),
            "chunk_count": 0,
            "upserted_count": 0,
            "status": "no_chunks",
        }

    print("Step 4 & 5: Embedding and upserting into Pinecone...")
    upserted_count = embed_and_upsert(chunks, source_name=pdf_path)

    print(f"\nDone! {len(chunks)} chunks from '{pdf_path}' are now in Pinecone.\n")
    return {
        "source": pdf_path,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "upserted_count": upserted_count,
        "status": "ok",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py path/to/file.pdf")
        sys.exit(1)

    ingest(sys.argv[1])
