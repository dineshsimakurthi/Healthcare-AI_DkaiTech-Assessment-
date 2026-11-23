"""
PART 2a - RAG INDEXING
documents -> clean -> chunks -> embeddings -> Chroma vector database

Run once:  python src/build_index.py
"""
import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
CHROMA_DIR = ROOT / "chroma_db"

# Free, small, local embedding model (no API key needed)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_and_clean():
    """1) Load every .txt file in docs/ and 2) clean the text."""
    docs = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"[ \t]+", " ", text).strip()          # tidy extra spaces
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    print(f"Loaded {len(docs)} documents")
    return docs


def split(docs):
    """3) Split into small overlapping chunks so each one covers one idea."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    return chunks


def build(embeddings=None):
    """4) Embed the chunks and 5) store them in Chroma."""
    embeddings = embeddings or get_embeddings()
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)          # rebuild from scratch
    chunks = split(load_and_clean())
    Chroma.from_documents(chunks, embeddings, persist_directory=str(CHROMA_DIR))
    print(f"Vector store saved to {CHROMA_DIR}")


if __name__ == "__main__":
    build()
