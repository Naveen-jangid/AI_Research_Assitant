"""
app.py – Streamlit frontend for the AI Research Assistant.

UI Layout
─────────
┌─────────────────────────────────────────────────────────────┐
│  Sidebar                    │  Main Panel                   │
│  ─────────────────────────  │  ─────────────────────────── │
│  • App title + settings     │  • Tab: Chat (RAG Q&A)        │
│  • PDF uploader             │  • Tab: Summary               │
│  • Processing status        │  • Tab: Methodology           │
│  • ─────────────────────    │  • Tab: Formulas              │
│  • [Generate Summary]       │                               │
│  • [Explain Methodology]    │                               │
│  • [Extract Formulas]       │                               │
│  • ─────────────────────    │                               │
│  • [Clear Session]          │                               │
└─────────────────────────────────────────────────────────────┘

Session state keys
──────────────────
  "documents"     : list[Document]  – chunked LangChain docs
  "pages"         : list[PageContent]
  "store"         : VectorStoreManager
  "pipeline"      : dict (qa, summary, methodology, formulas)
  "chat_history"  : list[{"role": str, "content": str}]
  "action_results": dict[str, str]  – cached one-click outputs
  "processed_files": set[str]       – deduplicate uploads
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import streamlit as st

from config import cfg
from document_processor import process_pdf, get_full_text, PageContent
from vector_store import VectorStoreManager
from rag_pipeline import build_pipeline, QAResponse, ActionResponse
from langchain.schema import Document

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ── Page configuration ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title=cfg.app_title,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Chat bubbles */
.user-bubble {
    background: #1e3a5f;
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    margin: 4px 0 4px auto;
    max-width: 80%;
    width: fit-content;
    margin-left: auto;
}
.assistant-bubble {
    background: #f0f4f8;
    color: #1a1a2e;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 16px;
    margin: 4px auto 4px 0;
    max-width: 85%;
    width: fit-content;
}
/* Citation boxes */
.citation-box {
    background: #e8f4fd;
    border-left: 4px solid #2196f3;
    padding: 8px 12px;
    margin: 6px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.82em;
    color: #444;
}
/* Section results */
.result-box {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 20px;
    margin-top: 12px;
}
/* Status indicator */
.status-ready  { color: #2e7d32; font-weight: 600; }
.status-idle   { color: #757575; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ───────────────────────────────────────────────

def _init_session():
    defaults = {
        "documents": [],
        "pages": [],
        "store": None,
        "pipeline": None,
        "chat_history": [],
        "action_results": {},
        "processed_files": set(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session()


# ── Config validation banner ───────────────────────────────────────────────────

def _check_config() -> bool:
    errors = cfg.validate()
    if errors:
        st.error(
            "**Configuration errors detected.** Please fix your `.env` file:\n\n"
            + "\n".join(f"- {e}" for e in errors)
        )
        return False
    return True


# ── PDF processing helpers ─────────────────────────────────────────────────────

def _process_uploaded_files(uploaded_files) -> None:
    """Process newly uploaded PDFs and update the vector store."""
    new_files = [
        f for f in uploaded_files
        if f.name not in st.session_state.processed_files
    ]
    if not new_files:
        return

    source_names = [f.name for f in uploaded_files]

    # Initialise (or reuse) vector store keyed to the full set of source files
    if st.session_state.store is None:
        st.session_state.store = VectorStoreManager(source_files=source_names)

    progress = st.sidebar.progress(0, text="Processing PDFs…")
    all_new_docs: list[Document] = []
    all_new_pages: list[PageContent] = []

    for i, uploaded_file in enumerate(new_files):
        progress.progress(
            int((i / len(new_files)) * 60),
            text=f"Parsing {uploaded_file.name}…"
        )
        pdf_bytes = uploaded_file.read()
        docs, pages = process_pdf(pdf_bytes, filename=uploaded_file.name)
        all_new_docs.extend(docs)
        all_new_pages.extend(pages)
        st.session_state.processed_files.add(uploaded_file.name)
        logger.info("Parsed '%s': %d pages, %d chunks", uploaded_file.name, len(pages), len(docs))

    # Update session lists
    st.session_state.documents.extend(all_new_docs)
    st.session_state.pages.extend(all_new_pages)

    progress.progress(70, text="Embedding chunks…")
    st.session_state.store.add_documents(all_new_docs)

    progress.progress(90, text="Building pipeline…")
    st.session_state.pipeline = build_pipeline(st.session_state.store)

    progress.progress(100, text="Done!")
    time.sleep(0.4)
    progress.empty()

    st.sidebar.success(
        f"✅ Indexed **{len(all_new_docs)}** chunks from "
        f"**{len(new_files)}** file(s)."
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        st.title("🔬 " + cfg.app_title)
        st.caption(f"Model: `{cfg.active_model_name}` | Embeddings: `{cfg.embedding_provider}`")

        st.divider()

        # ── File uploader ─────────────────────────────────────────────────────
        st.subheader("📄 Upload Research Papers")
        uploaded_files = st.file_uploader(
            label="Drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Maximum file size: {cfg.max_file_size_mb} MB per file",
            label_visibility="collapsed",
        )

        if uploaded_files:
            _process_uploaded_files(uploaded_files)

        # ── Status indicator ──────────────────────────────────────────────────
        st.divider()
        if st.session_state.store and st.session_state.store.is_ready:
            doc_count = len(st.session_state.documents)
            page_count = len(st.session_state.pages)
            file_count = len(st.session_state.processed_files)
            st.markdown(
                f'<span class="status-ready">● Ready</span>&nbsp;'
                f'{file_count} file(s) · {page_count} pages · {doc_count} chunks',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-idle">● Idle</span> Upload a PDF to begin.',
                unsafe_allow_html=True,
            )

        # ── One-click action buttons ──────────────────────────────────────────
        st.divider()
        st.subheader("⚡ Quick Actions")

        pipeline_ready = st.session_state.pipeline is not None
        pages_available = bool(st.session_state.pages)

        col_a, col_b = st.columns(2)
        with col_a:
            summary_btn = st.button(
                "📋 Summary",
                disabled=not pages_available,
                use_container_width=True,
                help="Generate a structured academic summary",
            )
        with col_b:
            methodology_btn = st.button(
                "🔬 Methodology",
                disabled=not pipeline_ready,
                use_container_width=True,
                help="Explain the methodology in plain English",
            )

        formula_btn = st.button(
            "∑ Extract Formulas",
            disabled=not pipeline_ready,
            use_container_width=True,
            help="Find and annotate mathematical formulas",
        )

        # ── Handle button clicks ──────────────────────────────────────────────
        if summary_btn and pages_available:
            with st.spinner("Generating summary…"):
                full_text = get_full_text(st.session_state.pages)
                result: ActionResponse = st.session_state.pipeline["summary"].generate(full_text)
                st.session_state.action_results["summary"] = result.content
            st.success("Summary ready! See the Summary tab →")

        if methodology_btn and pipeline_ready:
            with st.spinner("Explaining methodology…"):
                result = st.session_state.pipeline["methodology"].explain()
                st.session_state.action_results["methodology"] = result.content
            st.success("Methodology ready! See the Methodology tab →")

        if formula_btn and pipeline_ready:
            with st.spinner("Extracting formulas…"):
                result = st.session_state.pipeline["formulas"].extract()
                st.session_state.action_results["formulas"] = result.content
            st.success("Formulas ready! See the Formulas tab →")

        # ── Clear session ─────────────────────────────────────────────────────
        st.divider()
        if st.button("🗑️ Clear Session", use_container_width=True):
            _clear_session()
            st.rerun()


def _clear_session():
    """Reset all session state."""
    for key in ["documents", "pages", "pipeline", "chat_history", "action_results"]:
        if key in st.session_state:
            st.session_state[key] = [] if isinstance(st.session_state[key], list) else (
                {} if isinstance(st.session_state[key], dict) else None
            )
    if st.session_state.get("store"):
        st.session_state.store.clear()
    st.session_state.store = None
    st.session_state.processed_files = set()


# ── Chat Tab ───────────────────────────────────────────────────────────────────

def _render_chat_tab():
    st.header("💬 Ask Questions About Your Paper")

    if not st.session_state.pipeline:
        st.info("👆 Upload a PDF in the sidebar to start chatting.")
        return

    # ── Chat history display ──────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for turn in st.session_state.chat_history:
            role = turn["role"]
            content = turn["content"]
            citations = turn.get("citations", [])

            if role == "user":
                st.markdown(
                    f'<div class="user-bubble">{content}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="assistant-bubble">{content}</div>',
                    unsafe_allow_html=True,
                )
                # Render citations
                if citations:
                    with st.expander(f"📎 {len(citations)} source(s) cited", expanded=False):
                        for cit in citations:
                            st.markdown(
                                f'<div class="citation-box">'
                                f'<strong>📄 {cit["source"]}</strong> · Page {cit["page"]}<br>'
                                f'<em>"{cit["excerpt"]}…"</em>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

    # ── Input box ─────────────────────────────────────────────────────────────
    st.divider()
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                label="Ask a question",
                placeholder="e.g. What is the impact of water-cement ratio on chloride migration?",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and user_input.strip():
        _handle_chat_message(user_input.strip())
        st.rerun()


def _handle_chat_message(question: str):
    """Process a user question and append Q&A to chat history."""
    # Add user turn immediately
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Retrieve & generate
    qa_chain = st.session_state.pipeline["qa"]
    response: QAResponse = qa_chain.ask(
        question=question,
        chat_history=st.session_state.chat_history[:-1],  # exclude the just-added user turn
    )

    # Serialise citations for storage
    serialised_citations = [
        {
            "source": c.source,
            "page": c.page,
            "chunk_index": c.chunk_index,
            "excerpt": c.excerpt,
        }
        for c in response.citations
    ]

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response.answer,
        "citations": serialised_citations,
    })


# ── Summary Tab ────────────────────────────────────────────────────────────────

def _render_summary_tab():
    st.header("📋 Paper Summary")
    if not st.session_state.pages:
        st.info("Upload a PDF and click **📋 Summary** in the sidebar.")
        return

    if "summary" not in st.session_state.action_results:
        st.info("Click **📋 Summary** in the sidebar to generate a structured summary.")
        # Offer inline trigger for convenience
        if st.button("Generate Summary Now"):
            with st.spinner("Generating summary…"):
                full_text = get_full_text(st.session_state.pages)
                result: ActionResponse = st.session_state.pipeline["summary"].generate(full_text)
                st.session_state.action_results["summary"] = result.content
            st.rerun()
        return

    st.markdown(
        f'<div class="result-box">{st.session_state.action_results["summary"]}</div>',
        unsafe_allow_html=True,
    )

    # Download button
    st.download_button(
        label="⬇️ Download Summary (Markdown)",
        data=st.session_state.action_results["summary"],
        file_name="summary.md",
        mime="text/markdown",
    )

    if st.button("🔄 Regenerate Summary"):
        del st.session_state.action_results["summary"]
        st.rerun()


# ── Methodology Tab ────────────────────────────────────────────────────────────

def _render_methodology_tab():
    st.header("🔬 Methodology Explained")
    if not st.session_state.pipeline:
        st.info("Upload a PDF and click **🔬 Methodology** in the sidebar.")
        return

    if "methodology" not in st.session_state.action_results:
        st.info("Click **🔬 Methodology** in the sidebar to explain the methodology.")
        if st.button("Explain Methodology Now"):
            with st.spinner("Explaining methodology…"):
                result: ActionResponse = st.session_state.pipeline["methodology"].explain()
                st.session_state.action_results["methodology"] = result.content
            st.rerun()
        return

    st.caption(
        "The methodology has been rewritten in plain English. "
        "Technical terms are defined in parentheses on first use."
    )
    st.markdown(
        f'<div class="result-box">{st.session_state.action_results["methodology"]}</div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        label="⬇️ Download Explanation (Markdown)",
        data=st.session_state.action_results["methodology"],
        file_name="methodology_explained.md",
        mime="text/markdown",
    )

    if st.button("🔄 Regenerate"):
        del st.session_state.action_results["methodology"]
        st.rerun()


# ── Formulas Tab ───────────────────────────────────────────────────────────────

def _render_formulas_tab():
    st.header("∑ Mathematical Formulas")
    if not st.session_state.pipeline:
        st.info("Upload a PDF and click **∑ Extract Formulas** in the sidebar.")
        return

    if "formulas" not in st.session_state.action_results:
        st.info("Click **∑ Extract Formulas** in the sidebar to find all equations.")
        if st.button("Extract Formulas Now"):
            with st.spinner("Scanning for formulas…"):
                result: ActionResponse = st.session_state.pipeline["formulas"].extract()
                st.session_state.action_results["formulas"] = result.content
            st.rerun()
        return

    st.caption(
        "Formulas are extracted from the paper with variable definitions and brief explanations."
    )

    # Display using markdown (formulas may contain LaTeX)
    formula_text = st.session_state.action_results["formulas"]
    st.markdown(formula_text)

    st.download_button(
        label="⬇️ Download Formulas (Markdown)",
        data=formula_text,
        file_name="formulas.md",
        mime="text/markdown",
    )

    if st.button("🔄 Re-extract"):
        del st.session_state.action_results["formulas"]
        st.rerun()


# ── Main Layout ────────────────────────────────────────────────────────────────

def main():
    if not _check_config():
        st.stop()

    _render_sidebar()

    # Main panel tabs
    tab_chat, tab_summary, tab_methodology, tab_formulas = st.tabs([
        "💬 Chat",
        "📋 Summary",
        "🔬 Methodology",
        "∑ Formulas",
    ])

    with tab_chat:
        _render_chat_tab()

    with tab_summary:
        _render_summary_tab()

    with tab_methodology:
        _render_methodology_tab()

    with tab_formulas:
        _render_formulas_tab()


if __name__ == "__main__":
    main()
