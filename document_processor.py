"""
document_processor.py – PDF ingestion, text extraction and smart chunking.

Design decisions:
  • PyMuPDF (fitz) is used as the primary parser because it preserves layout
    information (bounding boxes, blocks, spans) which is essential for handling
    multi-column formats common in IEEE/ACM papers.
  • Text extraction uses the "blocks" API so we can detect and re-order columns
    by reading left-column blocks before right-column blocks.
  • RecursiveCharacterTextSplitter respects paragraph/sentence boundaries to
    keep semantically coherent chunks.
  • Each chunk carries rich metadata so the UI can cite exact page numbers and
    source files.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import cfg


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PageContent:
    """Raw text + metadata for a single PDF page."""
    page_number: int          # 1-based
    text: str
    source: str               # filename
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


# ── Core PDF parser ───────────────────────────────────────────────────────────

class PDFParser:
    """
    Extracts text from PDFs while handling multi-column research paper layouts.

    Multi-column strategy:
      IEEE / ACM papers typically use a 2-column layout. We detect columns by
      checking whether text blocks are horizontally separated by more than
      COLUMN_GAP_RATIO * page_width, then sort blocks left-column-first so the
      extracted text reads in the correct reading order.
    """

    # If a gap between two text blocks exceeds this fraction of page width,
    # we treat them as separate columns.
    COLUMN_GAP_RATIO = 0.35

    def __init__(self, source_name: str):
        self.source_name = source_name

    # ── Public API ────────────────────────────────────────────────────────────

    def parse_bytes(self, pdf_bytes: bytes) -> list[PageContent]:
        """Parse a PDF supplied as raw bytes (e.g. from Streamlit uploader)."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return self._extract_pages(doc)

    def parse_file(self, path: str | Path) -> list[PageContent]:
        """Parse a PDF from disk."""
        doc = fitz.open(str(path))
        return self._extract_pages(doc)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_pages(self, doc: fitz.Document) -> list[PageContent]:
        pages: list[PageContent] = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = self._extract_page_text(page)
            text = self._clean_text(text)
            if text.strip():  # skip blank/image-only pages
                pages.append(PageContent(
                    page_number=page_index + 1,
                    text=text,
                    source=self.source_name,
                ))
        doc.close()
        return pages

    def _extract_page_text(self, page: fitz.Page) -> str:
        """
        Extract text in correct reading order, handling multi-column layouts.

        fitz block format: (x0, y0, x1, y1, text, block_no, block_type)
        block_type 0 = text, 1 = image
        """
        page_width = page.rect.width
        blocks = page.get_text("blocks")

        # Keep only text blocks
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

        if not text_blocks:
            return ""

        # Detect if page has a multi-column layout
        if self._is_multicolumn(text_blocks, page_width):
            text_blocks = self._sort_multicolumn(text_blocks, page_width)
        else:
            # Single-column: sort top-to-bottom
            text_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))

        return "\n\n".join(b[4] for b in text_blocks)

    def _is_multicolumn(self, blocks: list, page_width: float) -> bool:
        """
        Heuristic: if blocks span two horizontally distinct halves of the page
        with a clear vertical gap in the middle, it's multi-column.
        """
        midpoint = page_width / 2
        left_blocks = [b for b in blocks if b[2] < midpoint + 20]
        right_blocks = [b for b in blocks if b[0] > midpoint - 20]
        # Both halves must have content for it to be multi-column
        return bool(left_blocks) and bool(right_blocks) and len(right_blocks) >= 2

    def _sort_multicolumn(self, blocks: list, page_width: float) -> list:
        """
        Sort blocks so left-column text comes before right-column text,
        preserving top-to-bottom order within each column.
        """
        midpoint = page_width / 2
        left = sorted(
            [b for b in blocks if (b[0] + b[2]) / 2 < midpoint],
            key=lambda b: b[1]
        )
        right = sorted(
            [b for b in blocks if (b[0] + b[2]) / 2 >= midpoint],
            key=lambda b: b[1]
        )
        return left + right

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Normalise unicode, collapse excessive whitespace, and remove
        soft-hyphen line breaks common in PDF text extraction.
        """
        # Normalise unicode to NFC form
        text = unicodedata.normalize("NFC", text)
        # Remove soft hyphens used for word-wrapping in PDFs
        text = text.replace("\u00ad", "")
        # Rejoin words broken across lines with a hyphen (e.g. "con-\ntext" -> "context")
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # Collapse multiple blank lines to a single blank line
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace
        return text.strip()


# ── Chunker ───────────────────────────────────────────────────────────────────

class DocumentChunker:
    """
    Converts a list of PageContent objects into LangChain Documents with
    fine-grained metadata, using RecursiveCharacterTextSplitter for
    semantically coherent chunks.

    Separator priority (highest → lowest):
      1. Double newlines     → paragraph breaks
      2. Single newline      → line breaks
      3. Sentence endings    → ". " / "? " / "! "
      4. Comma-space         → clause boundaries
      5. Space               → last resort word boundary
      6. Empty string        → hard character split (never ideal)
    """

    SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""]

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or cfg.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_pages(self, pages: list[PageContent]) -> list[Document]:
        """
        Chunk a list of pages into LangChain Documents.

        Each Document carries:
          - page_content : the chunk text
          - metadata.source       : original filename
          - metadata.page         : page number (1-based)
          - metadata.chunk_index  : position within the page's chunks
          - metadata.total_chars  : character count of original page
        """
        documents: list[Document] = []

        for page in pages:
            page_chunks = self._splitter.split_text(page.text)
            for idx, chunk_text in enumerate(page_chunks):
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": page.source,
                        "page": page.page_number,
                        "chunk_index": idx,
                        "total_chars": page.char_count,
                    },
                )
                documents.append(doc)

        return documents


# ── High-level convenience function ───────────────────────────────────────────

def process_pdf(
    pdf_bytes: bytes,
    filename: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> tuple[list[Document], list[PageContent]]:
    """
    Full pipeline: bytes → parsed pages → chunked LangChain Documents.

    Returns
    -------
    documents : list[Document]
        Ready to be embedded and stored in the vector store.
    pages : list[PageContent]
        Raw per-page text, useful for full-document operations like
        summary generation or formula extraction.
    """
    parser = PDFParser(source_name=filename)
    pages = parser.parse_bytes(pdf_bytes)

    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = chunker.chunk_pages(pages)

    return documents, pages


def get_full_text(pages: list[PageContent]) -> str:
    """Concatenate all page texts into a single string (for full-doc LLM calls)."""
    return "\n\n".join(
        f"[Page {p.page_number}]\n{p.text}" for p in pages
    )
