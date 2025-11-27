# local_ingest_single.py
import os
from pathlib import Path
from typing import Optional
from docx import Document
from pypdf import PdfReader

from chromadb.config import Settings
from chromadb import Client
from sentence_transformers import SentenceTransformer
import uuid

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

# Chroma client (same duckdb+parquet format currently used)
chroma_client = Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(ROOT_DIR / ".chroma")
    )
)

# Memory/knowledge collection
COLLECTION_NAME = "sofia_memory"
collection = chroma_client.get_or_create_collection(
    COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# Local embedder (MiniLM)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def read_file_to_text(path: Path) -> Optional[str]:
    ext = path.suffix.lower()

    if ext == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs).strip()

    if ext == ".pdf":
        reader = PdfReader(str(path))
        chunks = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            chunks.append(txt.strip())
        return "\n".join(chunks).strip()

    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore").strip()

    # Skip unsupported types for now
    return None


def embed_and_store(path: Path):
    text = read_file_to_text(path)
    if not text:
        print(f"[WATCH] {path.name}: empty or unsupported format, skipping.")
        return

    # Embed the whole text (could be chunked later)
    vec = embedder.encode([text]).tolist()

    doc_id = f"{path.name}-{uuid.uuid4().hex[:8]}"
    metadata = {
        "filename": path.name,
        "source_path": str(path),
    }

    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=vec,
        metadatas=[metadata],
    )

    chroma_client.persist()  # flush to disk immediately
    print(f"[WATCH] {path.name} -> added to Chroma as {doc_id}")
