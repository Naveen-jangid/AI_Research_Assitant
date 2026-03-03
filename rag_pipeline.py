"""
rag_pipeline.py – RAG retrieval logic and specialised LLM prompt chains.

Four capabilities are implemented here:

  1. RAGQAChain         – Conversational Q&A with per-chunk citations.
  2. SummaryChain       – Structured academic summary (Abstract / Findings /
                          Limitations).
  3. MethodologyChain   – Jargon-free step-by-step methodology explanation.
  4. FormulaChain       – Mathematical formula extraction and annotation.

Prompt engineering principles:
  • Each system prompt defines a strict expert persona so the LLM stays on-task.
  • Structured output sections (e.g. "Key Findings:", "Step 1:") are enforced
    via prompt templates to make UI parsing predictable.
  • For QA, the LLM is explicitly instructed to cite page numbers so users can
    verify claims against the original paper.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from langchain.schema import Document, HumanMessage, SystemMessage

from config import cfg
from llm_handler import get_llm
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# ── Response data models ──────────────────────────────────────────────────────

@dataclass
class Citation:
    """Source reference attached to an answer chunk."""
    source: str
    page: int
    chunk_index: int
    excerpt: str   # first 200 chars of the chunk for context


@dataclass
class QAResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class ActionResponse:
    """Generic response for one-click action buttons."""
    content: str
    action: str   # "summary" | "methodology" | "formulas"


# ── System Prompts ────────────────────────────────────────────────────────────

_QA_SYSTEM_PROMPT = """You are an expert academic research assistant with deep \
knowledge across science, engineering, and mathematics. Your job is to answer \
questions about research papers accurately and concisely.

INSTRUCTIONS:
- Answer ONLY based on the provided context passages.
- If the context does not contain enough information, say "The paper does not \
  discuss this in the provided excerpts."
- For every factual claim, append a citation in the format [Page X] where X is \
  the page number from the context metadata.
- Use technical language appropriate for a graduate-level audience.
- Keep answers focused and avoid padding.

CONTEXT:
{context}
"""

_SUMMARY_SYSTEM_PROMPT = """You are an expert academic summariser. Your task is \
to produce a structured, high-quality summary of a research paper from the \
provided excerpts.

Output EXACTLY the following sections with these headings:

## Overview
One paragraph (3-5 sentences) describing what the paper is about, its goals,
and why it matters.

## Key Findings
A bullet-point list (4-8 items) of the most important results and conclusions.

## Methodology (Brief)
2-3 sentences describing the core experimental or analytical approach.

## Limitations & Future Work
A bullet-point list of stated or implied limitations and suggested future
directions.

Use formal academic language. Do not hallucinate — base everything on the text.
"""

_METHODOLOGY_SYSTEM_PROMPT = """You are a science communicator who specialises \
in making complex research accessible to intelligent non-specialists (e.g., \
undergraduate students or curious professionals).

Your task: Explain the methodology of the research paper in clear, simple, \
step-by-step terms.

RULES:
- Replace jargon with plain English. When a technical term is unavoidable,
  define it in parentheses on first use.
- Use numbered steps (Step 1, Step 2, …) to describe the experimental process.
- Include a short "Why this matters" note after each step, explaining its
  purpose in the larger experiment.
- End with a brief "Summary" section restating the overall approach in 2-3
  plain-English sentences.
- Do NOT make up details not present in the context.

CONTEXT:
{context}
"""

_FORMULA_SYSTEM_PROMPT = """You are a mathematical notation expert. Analyse the \
following text extracted from a research paper and extract ALL mathematical \
formulas, equations, and expressions.

For each formula found, output in this EXACT format:

**Formula N:** <formula in LaTeX or plain text>
**Description:** <one-sentence explanation of what this formula represents>
**Variables:** <define each variable/symbol used>
---

Rules:
- If the text contains inline formulas (e.g., "chloride diffusion coefficient D"),
  extract them too.
- Number formulas sequentially starting from 1.
- If the paper labels equations (e.g., "Equation 3" or "(3)"), preserve that label.
- If NO formulas are found, respond with: "No mathematical formulas were detected
  in the provided excerpts."

TEXT TO ANALYSE:
{text}
"""


# ── Helper: format retrieved docs into context string ─────────────────────────

def _format_context(docs: list[Document]) -> tuple[str, list[Citation]]:
    """
    Convert retrieved Documents into a context string and Citation objects.

    Returns
    -------
    context_str : str
        Formatted context block passed to the LLM.
    citations : list[Citation]
        Structured citation data for the UI.
    """
    context_parts: list[str] = []
    citations: list[Citation] = []

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        source = meta.get("source", "Unknown")
        page = meta.get("page", 0)
        chunk_idx = meta.get("chunk_index", 0)

        context_parts.append(
            f"[Passage {i} | Source: {source} | Page: {page}]\n{doc.page_content}"
        )
        citations.append(Citation(
            source=source,
            page=page,
            chunk_index=chunk_idx,
            excerpt=doc.page_content[:200],
        ))

    return "\n\n---\n\n".join(context_parts), citations


# ── 1. RAG Q&A Chain ──────────────────────────────────────────────────────────

class RAGQAChain:
    """
    Retrieval-Augmented Generation for conversational Q&A.

    Usage
    -----
    chain = RAGQAChain(vector_store_manager)
    result = chain.ask("What is the chloride migration coefficient?")
    print(result.answer)
    for c in result.citations:
        print(f"  → Page {c.page}: {c.excerpt[:80]}…")
    """

    def __init__(
        self,
        store: VectorStoreManager,
        k: Optional[int] = None,
    ):
        self._store = store
        self._k = k or cfg.top_k_retrieval
        self._llm = get_llm()

    def ask(
        self,
        question: str,
        chat_history: Optional[list[dict]] = None,
    ) -> QAResponse:
        """
        Answer a question using retrieved context.

        Parameters
        ----------
        question : str
            The user's question.
        chat_history : list of {"role": str, "content": str}, optional
            Prior conversation turns for multi-turn context.
        """
        if not self._store.is_ready:
            return QAResponse(
                answer="Please upload and process a PDF document first.",
                citations=[],
            )

        # Retrieve relevant chunks
        docs = self._store.similarity_search(question, k=self._k)
        context_str, citations = _format_context(docs)

        # Build message list
        messages = [SystemMessage(content=_QA_SYSTEM_PROMPT.format(context=context_str))]

        # Inject prior conversation turns (last 6 turns max to stay within context)
        if chat_history:
            for turn in chat_history[-6:]:
                if turn["role"] == "user":
                    messages.append(HumanMessage(content=turn["content"]))
                else:
                    from langchain.schema import AIMessage
                    messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=question))

        response = self._llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        return QAResponse(answer=answer, citations=citations)


# ── 2. Summary Chain ──────────────────────────────────────────────────────────

class SummaryChain:
    """
    Generate a structured academic summary of the entire paper.

    Because the full paper may exceed the LLM's context window, we use a
    map-reduce approach:
      • Map   : summarise each page independently (for long docs)
      • Reduce: combine page summaries into the final structured output

    For papers that fit in one context window (< ~40 pages), we send all
    text directly.
    """

    # Rough character limit before switching to map-reduce
    DIRECT_LIMIT = 80_000

    def __init__(self):
        self._llm = get_llm()

    def generate(self, full_text: str) -> ActionResponse:
        """
        Parameters
        ----------
        full_text : str
            Concatenated text from all pages (output of get_full_text()).
        """
        logger.info("Generating summary. Text length: %d chars", len(full_text))

        if len(full_text) <= self.DIRECT_LIMIT:
            content = self._direct_summary(full_text)
        else:
            content = self._mapreduce_summary(full_text)

        return ActionResponse(content=content, action="summary")

    def _direct_summary(self, text: str) -> str:
        messages = [
            SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=f"Please summarise the following research paper:\n\n{text}"),
        ]
        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    def _mapreduce_summary(self, text: str) -> str:
        """Split text into chunks, summarise each, then combine."""
        chunk_size = self.DIRECT_LIMIT
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        # Map phase: summarise each chunk
        partial_summaries: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            logger.info("Map phase: summarising chunk %d/%d", idx, len(chunks))
            messages = [
                SystemMessage(content=(
                    "Summarise the following excerpt from a research paper in 3-5 sentences, "
                    "focusing on key findings and methodology. Be concise."
                )),
                HumanMessage(content=chunk),
            ]
            resp = self._llm.invoke(messages)
            partial_summaries.append(resp.content)

        # Reduce phase: combine partial summaries into structured output
        combined = "\n\n---\n\n".join(partial_summaries)
        messages = [
            SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=(
                "Based on the following partial summaries of a research paper, "
                "produce the final structured summary:\n\n" + combined
            )),
        ]
        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)


# ── 3. Methodology Chain ──────────────────────────────────────────────────────

class MethodologyChain:
    """
    Retrieve methodology-related sections and explain them in plain English.

    The retrieval step uses a battery of methodology-related queries to find
    the most relevant chunks regardless of how the section is titled in the paper
    (e.g. "Materials and Methods", "Experimental Setup", "Procedure").
    """

    METHODOLOGY_QUERIES = [
        "experimental setup methodology materials methods procedure",
        "sample preparation experimental design protocol",
        "measurement procedure testing conditions specimens",
        "data collection analysis statistical methods",
    ]

    def __init__(self, store: VectorStoreManager):
        self._store = store
        self._llm = get_llm()

    def explain(self) -> ActionResponse:
        """Retrieve and explain the methodology sections."""
        if not self._store.is_ready:
            return ActionResponse(
                content="Please upload and process a PDF document first.",
                action="methodology",
            )

        # Aggregate unique chunks from multiple targeted queries
        seen_excerpts: set[str] = set()
        methodology_docs: list[Document] = []

        for query in self.METHODOLOGY_QUERIES:
            docs = self._store.similarity_search(query, k=3)
            for doc in docs:
                key = doc.page_content[:100]
                if key not in seen_excerpts:
                    seen_excerpts.add(key)
                    methodology_docs.append(doc)

        if not methodology_docs:
            return ActionResponse(
                content="Could not find methodology-related sections in the document.",
                action="methodology",
            )

        context_str, _ = _format_context(methodology_docs)

        messages = [
            SystemMessage(content=_METHODOLOGY_SYSTEM_PROMPT.format(context=context_str)),
            HumanMessage(content=(
                "Explain the methodology of this research paper step-by-step in simple terms."
            )),
        ]
        response = self._llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        return ActionResponse(content=content, action="methodology")


# ── 4. Formula Extraction Chain ───────────────────────────────────────────────

class FormulaChain:
    """
    Scan the document for mathematical formulas and explain each one.

    Strategy:
      • First, search for chunks containing mathematical indicators (=, ∑, ∫,
        Greek letters spelled out, equation/formula keywords).
      • Additionally, do a broad retrieval pass to catch inline equations.
      • Pass all candidate text to the LLM with the formula extraction prompt.
    """

    # Regex patterns to pre-filter text likely containing math
    MATH_PATTERNS = [
        r"=\s*[\w\d\(\)\[\]\{\}]+",          # any assignment/equation
        r"[∑∫∂∇×·÷±√∞≤≥≠≈∝∈⊆]",           # math symbols
        r"\b(equation|formula|expression|coefficient|parameter)\b",
        r"\b(alpha|beta|gamma|delta|epsilon|lambda|sigma|omega|theta|phi)\b",
        r"\^[\d\w]+",                          # exponents (x^2)
        r"_[\d\w]+",                           # subscripts (C_0)
    ]

    FORMULA_QUERIES = [
        "mathematical equation formula coefficient expression",
        "equation diffusion concentration gradient model",
        "parameter calculation statistical regression model fit",
        "Fick diffusion law permeability coefficient formula",
    ]

    def __init__(self, store: VectorStoreManager):
        self._store = store
        self._llm = get_llm()
        self._math_re = re.compile(
            "|".join(self.MATH_PATTERNS), re.IGNORECASE
        )

    def extract(self) -> ActionResponse:
        """Retrieve and extract all mathematical formulas from the document."""
        if not self._store.is_ready:
            return ActionResponse(
                content="Please upload and process a PDF document first.",
                action="formulas",
            )

        # Targeted retrieval for formula-heavy chunks
        seen: set[str] = set()
        formula_docs: list[Document] = []

        for query in self.FORMULA_QUERIES:
            docs = self._store.similarity_search(query, k=4)
            for doc in docs:
                key = doc.page_content[:100]
                if key not in seen and self._math_re.search(doc.page_content):
                    seen.add(key)
                    formula_docs.append(doc)

        # Fallback: if no math-filtered docs found, use all retrieved
        if not formula_docs:
            for query in self.FORMULA_QUERIES[:2]:
                docs = self._store.similarity_search(query, k=3)
                for doc in docs:
                    key = doc.page_content[:100]
                    if key not in seen:
                        seen.add(key)
                        formula_docs.append(doc)

        combined_text = "\n\n".join(doc.page_content for doc in formula_docs)

        messages = [
            SystemMessage(content="You are a mathematical notation expert."),
            HumanMessage(content=_FORMULA_SYSTEM_PROMPT.format(text=combined_text)),
        ]
        response = self._llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        return ActionResponse(content=content, action="formulas")


# ── Convenience factory ───────────────────────────────────────────────────────

def build_pipeline(store: VectorStoreManager) -> dict:
    """
    Build all pipeline objects at once.  Call this once after indexing
    and store the result in Streamlit session state.

    Returns
    -------
    dict with keys: "qa", "summary", "methodology", "formulas"
    """
    return {
        "qa": RAGQAChain(store),
        "summary": SummaryChain(),
        "methodology": MethodologyChain(store),
        "formulas": FormulaChain(store),
    }
