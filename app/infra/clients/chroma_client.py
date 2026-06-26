from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app.core.settings import (
    RAG_CHROMA_COLLECTION,
    RAG_CHROMA_PERSIST_DIR,
    RAG_EMBEDDING_BASE_URL,
    RAG_EMBEDDING_MODEL,
)


def _build_chroma_where_filter(metadata_filter: dict[str, Any] | None) -> dict[str, Any] | None:
    """将 metadata_filter 字典转换为 Chroma $and 查询语法，过滤掉值为 None 的字段。"""
    if not metadata_filter:
        return None

    items = [(key, value) for key, value in metadata_filter.items() if value is not None]
    if not items:
        return None

    # 使用 $and 显式运算符以兼容新版 Chroma where 语法
    return {"$and": [{key: value} for key, value in items]}


@lru_cache(maxsize=1)
def get_chroma_store(collection_name: str = RAG_CHROMA_COLLECTION) -> Chroma:
    """初始化 Chroma 向量库并缓存为单例，避免重复加载 embedding 模型。"""
    embedding = OllamaEmbeddings(
        model=RAG_EMBEDDING_MODEL,
        base_url=RAG_EMBEDDING_BASE_URL,
    )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=RAG_CHROMA_PERSIST_DIR,
    )


def search_chroma(
    *,
    query: str,
    top_k: int,
    collection_name: str = RAG_CHROMA_COLLECTION,
    metadata_filter: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """在 Chroma 向量库中执行相似度检索，返回 top_k 条文档及对应相关性分数。"""
    store = get_chroma_store(collection_name)
    where_filter = _build_chroma_where_filter(metadata_filter)
    return store.similarity_search_with_relevance_scores(
        query=query,
        k=top_k,
        filter=where_filter,
    )


def build_citations(docs_with_scores: list[tuple[Document, float]]) -> list[dict[str, Any]]:
    """将检索结果整理为引用列表，提取来源、领域、书籍ID、分块索引、分数及内容预览。"""
    citations: list[dict[str, Any]] = []
    for doc, score in docs_with_scores:
        meta = doc.metadata or {}
        citations.append(
            {
                "source": meta.get("source", "unknown"),
                "domain": meta.get("domain", "unknown"),
                "book_id": meta.get("book_id", "unknown"),
                "chunk": meta.get("chunk_index", -1),
                "score": float(score),
                "preview": doc.page_content[:160],
            }
        )
    return citations