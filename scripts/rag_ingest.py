from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}

DEFAULT_OLLAMA_BASE_URL = "http://localhost:8001"
DEFAULT_CHROMA_COLLECTION = "rag_default"
DEFAULT_CHROMA_PERSIST_DIR = "./.chroma"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_KNOWLEDGE_DOMAIN = "general"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# rag的数据准备阶段
# 文档清洗 -> 切片（Chunking）-> 向量化（Embedding）-> 存入向量数据库

def _load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def _build_chunks(
    path: Path,
    *,
    knowledge_domain: str,
    book_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    text = _load_text(path)
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    source_doc = Document(
        page_content=text,
        metadata={
            "source": path.name,
            "path": str(path),
            "domain": knowledge_domain,
            "book_id": book_id,
        },
    )
    split_docs = splitter.split_documents([source_doc])

    chunks: list[Document] = []
    for idx, doc in enumerate(split_docs):
        metadata = dict(doc.metadata)
        metadata["chunk_index"] = idx
        chunks.append(Document(page_content=doc.page_content, metadata=metadata))
    return chunks


def _resolve_book_id(path: Path, custom_book_id: str | None) -> str:
    if custom_book_id:
        return custom_book_id
    return path.stem


def _get_store(
    *,
    collection_name: str,
    embedding_model: str,
    embedding_base_url: str,
    persist_directory: str,
) -> Chroma:
    embedding = OllamaEmbeddings(
        model=embedding_model,
        base_url=embedding_base_url,
    )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_directory,
    )


def _iter_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError("Input file type is not supported. Use .txt/.md/.pdf")
        return [input_path]

    files: list[Path] = []
    for path in input_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return sorted(files)


async def _ingest_files(
    paths: list[Path],
    collection_name: str,
    knowledge_domain: str,
    book_id: str | None,
    embedding_model: str,
    embedding_base_url: str,
    persist_directory: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    store = _get_store(
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        persist_directory=persist_directory,
    )
    total_chunks = 0
    total_inserted = 0

    for path in paths:
        resolved_book_id = _resolve_book_id(path, book_id)
        chunks = await asyncio.to_thread(
            _build_chunks,
            path,
            knowledge_domain=knowledge_domain,
            book_id=resolved_book_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunk_count = len(chunks)
        if chunk_count == 0:
            inserted_count = 0
            print(f"[SKIP] {path}: empty content")
        else:
            inserted_ids = await asyncio.to_thread(store.add_documents, chunks)
            inserted_count = len(inserted_ids)

        total_chunks += chunk_count
        total_inserted += inserted_count
        print(
            f"[OK] {path}: chunk_count={chunk_count}, inserted_count={inserted_count}, "
            f"collection={collection_name}, "
            f"domain={knowledge_domain}, book_id={resolved_book_id}"
        )

    print(
        f"Done. files={len(paths)}, total_chunks={total_chunks}, total_inserted={total_inserted}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ingest files into Chroma for RAG")
    parser.add_argument("--input", required=True, help="A file or a directory path")
    parser.add_argument(
        "--collection",
        default=os.getenv("RAG_CHROMA_COLLECTION", DEFAULT_CHROMA_COLLECTION),
        help="Chroma collection name",
    )
    parser.add_argument(
        "--domain",
        default=os.getenv("RAG_DEFAULT_KNOWLEDGE_DOMAIN", DEFAULT_KNOWLEDGE_DOMAIN),
        help="Knowledge domain metadata for retrieval filtering",
    )
    parser.add_argument(
        "--book-id",
        default=None,
        help="Optional book id metadata. If omitted, each file uses its stem as book_id",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="Embedding model name for Ollama",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=(
            os.getenv("RAG_EMBEDDING_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or DEFAULT_OLLAMA_BASE_URL
        ),
        help="Ollama base URL used by embedding model",
    )
    parser.add_argument(
        "--persist-dir",
        default=os.getenv("RAG_CHROMA_PERSIST_DIR", DEFAULT_CHROMA_PERSIST_DIR),
        help="Chroma persist directory",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("RAG_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
        help="Chunk size for text splitting",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("RAG_CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))),
        help="Chunk overlap for text splitting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    paths = _iter_files(input_path)
    if not paths:
        raise ValueError("No supported files found. Use .txt/.md/.pdf")

    asyncio.run(
        _ingest_files(
            paths,
            args.collection,
            args.domain,
            args.book_id,
            args.embedding_model,
            args.embedding_base_url,
            args.persist_dir,
            args.chunk_size,
            args.chunk_overlap,
        )
    )


if __name__ == "__main__":
    main()
