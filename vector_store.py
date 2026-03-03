"""
vector_store.py – Abstraction layer over ChromaDB (default) and FAISS.

Why this abstraction?
  • Different projects need different backends: ChromaDB persists to disk and
    supports filtering; FAISS is blazing fast for pure similarity search.
  • The rest of the codebase depends only on VectorStoreManager, never on a
    specific backend, so swapping is a one-line config change.

ChromaDB collection strategy:
  • One collection per session / per uploaded file set, keyed by a hash of
    the source filenames. This prevents cross-session contamination without
    requiring a database wipe.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from langchain.schema import Document
from langchain_community.vectorstores import Chroma, FAISS

from config import cfg
from llm_handler import get_embeddings

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collection_name(source_files: list[str]) -> str:
    """
    Generate a stable, URL-safe collection name from the set of source filenames.
    Chroma collection names must be 3-63 chars, alphanumeric + hyphens only.
    """
    key = "_".join(sorted(source_files))
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    return f"research-{digest}"


# ── Vector Store Manager ──────────────────────────────────────────────────────

class VectorStoreManager:
    """
    Manages creation, population, and querying of the vector store.

    Usage
    -----
    manager = VectorStoreManager()
    manager.add_documents(documents)          # embed and persist
    results = manager.similarity_search(query, k=5)
    """

    def __init__(self, source_files: Optional[list[str]] = None):
        self._embeddings = get_embeddings()
        self._store_type = cfg.vector_store_type  # "chroma" | "faiss"
        self._collection_name = _collection_name(source_files or ["default"])
        self._store: Optional[Chroma | FAISS] = None

    # ── Population ────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> None:
        """
        Embed documents and upsert them into the vector store.
        Calling this multiple times is safe – documents are appended.
        """
        if not documents:
            logger.warning("add_documents called with empty list; skipping.")
            return

        logger.info(
            "Embedding %d chunks into %s collection '%s'",
            len(documents), self._store_type, self._collection_name
        )

        if self._store_type == "chroma":
            self._store = self._upsert_chroma(documents)
        elif self._store_type == "faiss":
            self._store = self._upsert_faiss(documents)
        else:
            raise ValueError(f"Unknown VECTOR_STORE_TYPE: '{self._store_type}'")

        logger.info("Vector store ready – %d documents indexed.", len(documents))

    def _upsert_chroma(self, documents: list[Document]) -> Chroma:
        """Create or extend a ChromaDB collection."""
        if self._store is not None and isinstance(self._store, Chroma):
            # Extend existing in-memory store
            self._store.add_documents(documents)
            return self._store

        return Chroma.from_documents(
            documents=documents,
            embedding=self._embeddings,
            collection_name=self._collection_name,
            persist_directory=cfg.chroma_persist_dir,
        )

    def _upsert_faiss(self, documents: list[Document]) -> FAISS:
        """Create or merge a FAISS index."""
        if self._store is not None and isinstance(self._store, FAISS):
            new_store = FAISS.from_documents(documents, self._embeddings)
            self._store.merge_from(new_store)
            return self._store

        return FAISS.from_documents(documents, self._embeddings)

    # ── Querying ──────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: float = 0.0,
    ) -> list[Document]:
        """
        Return the top-k most relevant documents for the query.

        Parameters
        ----------
        query : str
            The user's question or search string.
        k : int, optional
            Number of results to return. Defaults to cfg.top_k_retrieval.
        score_threshold : float
            Minimum similarity score (0.0 = no filter). Only applied when the
            backend supports score filtering (ChromaDB).
        """
        if self._store is None:
            raise RuntimeError("Vector store is empty. Upload and process a PDF first.")

        k = k or cfg.top_k_retrieval

        if self._store_type == "chroma" and score_threshold > 0.0:
            # ChromaDB supports score-filtered search
            results = self._store.similarity_search_with_relevance_scores(query, k=k)
            return [doc for doc, score in results if score >= score_threshold]

        return self._store.similarity_search(query, k=k)

    def similarity_search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """Like similarity_search but also returns relevance scores."""
        if self._store is None:
            raise RuntimeError("Vector store is empty. Upload and process a PDF first.")

        k = k or cfg.top_k_retrieval

        if self._store_type == "chroma":
            return self._store.similarity_search_with_relevance_scores(query, k=k)
        elif self._store_type == "faiss":
            return self._store.similarity_search_with_score(query, k=k)

        return [(doc, 1.0) for doc in self._store.similarity_search(query, k=k)]

    # ── Retriever interface ───────────────────────────────────────────────────

    def as_retriever(self, k: Optional[int] = None):
        """
        Return a LangChain-compatible retriever object for use in chains.
        """
        if self._store is None:
            raise RuntimeError("Vector store is empty.")
        k = k or cfg.top_k_retrieval
        return self._store.as_retriever(search_kwargs={"k": k})

    # ── State checks ──────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if the store has been populated with at least one document."""
        return self._store is not None

    def clear(self) -> None:
        """
        Drop all documents (useful for the 'Clear Session' button in the UI).
        Note: for ChromaDB this only clears the in-memory reference; the
        on-disk collection is preserved for the next session.
        """
        self._store = None
        logger.info("Vector store reference cleared.")
