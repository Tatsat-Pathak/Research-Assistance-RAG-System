"""
app.py  —  Research RAG Assistant · Streamlit UI
=================================================
Run with:
    streamlit run app.py
"""

import os
import time
import uuid
import shutil
import traceback
import streamlit as st
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# ─────────────────────────────────────────────────────────────────────────────
# DIRECT IMPORTS FROM RAG_Pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
import RAG_Pipeline
from RAG_Pipeline import (
    parse_document,
    chunks_generation,
    combining_chunks,
    load_embedded_model,
    create_embedding_text,
    initialize_vector_database,
    initialize_collections,
    query_builder,
    answer_generator,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES  (UI contract)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Source:
    """One retrieved source displayed under an assistant message."""
    doc_title:       str
    section_heading: str
    similarity:      float
    is_table:        bool = False
    snippet:         str  = ""

@dataclass
class AskResult:
    """Return type of ask()."""
    answer:  str
    sources: List[Source] = field(default_factory=list)
    error:   Optional[str] = None

@dataclass
class IndexResult:
    """Return type of index_pdfs()."""
    success:   bool
    doc_count: int = 0
    message:   str = ""


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETONS  (loaded once, reused across Streamlit reruns)
# ─────────────────────────────────────────────────────────────────────────────

_embedding_model = None
_chroma_client   = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = load_embedded_model("BAAI/bge-base-en-v1.5")
    return _embedding_model

def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = initialize_vector_database()
    return _chroma_client


# ─────────────────────────────────────────────────────────────────────────────
# index_pdfs  —  calls RAG_Pipeline functions
# ─────────────────────────────────────────────────────────────────────────────

def index_pdfs(pdf_paths: List[str]) -> IndexResult:
    try:
        # 1. Parse every PDF
        papers_section_heading: Dict[str, list] = {}
        parsed_count = 0
        for path in pdf_paths:
            file_name, _, sections = parse_document(path)
            if sections:
                papers_section_heading[file_name] = sections
                parsed_count += 1

        if not papers_section_heading:
            return IndexResult(success=False, doc_count=0,
                               message="No readable sections found in the uploaded PDFs.")

        # 2. Generate chunks
        paper_chunks = chunks_generation(papers_section_heading, chunk_size=500)

        # 3. Flatten
        all_chunks, all_ids = combining_chunks(paper_chunks)
        if not all_chunks:
            return IndexResult(success=False, doc_count=parsed_count,
                               message="Documents parsed but no chunks were generated.")

        # 4. Embed
        model = _get_embedding_model()
        RAG_Pipeline.embedding_model = model          # inject for query_builder()
        embeddings, shape = create_embedding_text(model, all_chunks, size=64)

        # 5. ChromaDB — drop all existing collections, recreate fresh
        client = _get_chroma_client()
        try:
            for col in client.list_collections():
                col_name = col.name if hasattr(col, 'name') else str(col)
                try:
                    client.delete_collection(col_name)
                except Exception:
                    pass
        except Exception:
            pass

        collection = initialize_collections(client)

        # 6. Store (bypasses add_in_collection typo: embedding → embeddings)
        collection.add(
            ids        = all_ids,
            documents  = all_chunks,
            embeddings = embeddings.tolist(),
        )

        RAG_Pipeline.collection = collection          # inject for query_builder()

        return IndexResult(
            success=True, doc_count=parsed_count,
            message=f"Indexed {parsed_count} document(s) → {len(all_chunks)} chunks stored.",
        )

    except Exception as exc:
        tb = traceback.format_exc()
        return IndexResult(success=False, doc_count=0,
                           message=f"Indexing error: {exc}\n\nTraceback:\n{tb}")


# ─────────────────────────────────────────────────────────────────────────────
# ask  —  calls RAG_Pipeline functions directly
# ─────────────────────────────────────────────────────────────────────────────

def ask(question: str, pdf_paths=None, chat_history=None) -> AskResult:
    try:
        if not hasattr(RAG_Pipeline, "collection") or RAG_Pipeline.collection is None:
            return AskResult(answer="",
                             error="No documents indexed yet. Please upload and index PDFs first.")

        # 1. Build prompt + retrieve context
        prompt, source_titles, chunk_ids = query_builder(question)

        # 2. Generate answer
        answer = answer_generator(prompt)
        if answer == "Error":
            return AskResult(answer="",
                             error="LLM API returned an error. Check your OPEN_ROUTER_API_KEY.")

        # 3. Build Source objects from chunk IDs
        sources: List[Source] = []
        seen_ids: set = set()
        for cid in chunk_ids:
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            pdf_marker = cid.find(".pdf")
            if pdf_marker != -1:
                doc_title = cid[:pdf_marker + 4]
                remainder = cid[pdf_marker + 5:]
            else:
                doc_title = cid
                remainder = ""

            chunk_marker = remainder.rfind("_chunk_")
            if chunk_marker != -1:
                before_chunk = remainder[:chunk_marker]
                last_underscore = before_chunk.rfind("_")
                section_heading = before_chunk[:last_underscore] if last_underscore != -1 else before_chunk
            else:
                section_heading = remainder

            section_heading = section_heading.replace("_", " ").strip()
            doc_title_clean = doc_title.replace("_", " ").replace(".pdf", "").strip()

            sources.append(Source(
                doc_title       = doc_title_clean if doc_title_clean else doc_title,
                section_heading = section_heading if section_heading else "N/A",
                similarity      = 0.0,
                is_table        = False,
                snippet         = "",
            ))

        return AskResult(answer=answer, sources=sources)

    except Exception as exc:
        return AskResult(answer="", error=f"Pipeline error: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research RAG Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_DIR = Path("uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — premium dark research theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ──────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
    min-height: 100vh;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e6edf3;
}

/* ── Main header ─────────────────────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #1a2332 0%, #162032 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
}

.main-header h1 {
    color: #e6edf3;
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.main-header p {
    color: #8b949e;
    margin: 6px 0 0 0;
    font-size: 0.92rem;
}

/* ── Chat messages ───────────────────────────────────────────────────────── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding: 8px 0;
}

.msg-user {
    display: flex;
    justify-content: flex-end;
    animation: slideInRight 0.3s ease;
}

.msg-assistant {
    display: flex;
    justify-content: flex-start;
    animation: slideInLeft 0.3s ease;
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px); }
    to   { opacity: 1; transform: translateX(0); }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0); }
}

.bubble-user {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #ffffff;
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    font-size: 0.93rem;
    line-height: 1.6;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.3);
}

.bubble-assistant {
    background: linear-gradient(135deg, #1c2333, #1a2332);
    border: 1px solid #30363d;
    color: #e6edf3;
    padding: 16px 20px;
    border-radius: 18px 18px 18px 4px;
    max-width: 82%;
    font-size: 0.93rem;
    line-height: 1.7;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.bubble-assistant p { color: #e6edf3; }
.bubble-assistant strong { color: #60a5fa; }
.bubble-assistant code {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 1px 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #79c0ff;
}

.msg-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    margin-top: 4px;
}

.avatar-user      { background: linear-gradient(135deg, #2563eb, #7c3aed); margin-left: 10px; }
.avatar-assistant { background: linear-gradient(135deg, #0f766e, #0891b2); margin-right: 10px; }

/* ── Sources card ────────────────────────────────────────────────────────── */
.sources-header {
    color: #8b949e;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 12px 0 8px 0;
}

.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 3px 4px 3px 0;
    font-size: 0.78rem;
    color: #8b949e;
    cursor: default;
    transition: border-color 0.2s, color 0.2s;
}

.source-chip:hover { border-color: #58a6ff; color: #58a6ff; }
.source-chip .sim  { color: #3fb950; font-weight: 600; }
.source-chip .tbl  { color: #f78166; }

/* ── PDF pill badges ─────────────────────────────────────────────────────── */
.pdf-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #1c2333;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}

.pdf-pill:hover { border-color: #58a6ff; }

.pdf-pill .pdf-icon { font-size: 1.2rem; }

.pdf-pill .pdf-name {
    color: #e6edf3;
    font-size: 0.83rem;
    font-weight: 500;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.pdf-pill .pdf-size {
    color: #6e7681;
    font-size: 0.75rem;
    white-space: nowrap;
}

/* ── Status badge ────────────────────────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}

.status-ready {
    background: rgba(63, 185, 80, 0.12);
    color: #3fb950;
    border: 1px solid rgba(63, 185, 80, 0.3);
}

.status-idle {
    background: rgba(110, 118, 129, 0.12);
    color: #6e7681;
    border: 1px solid rgba(110, 118, 129, 0.3);
}

.status-indexing {
    background: rgba(210, 153, 34, 0.12);
    color: #d29922;
    border: 1px solid rgba(210, 153, 34, 0.3);
}

/* ── Thinking animation ──────────────────────────────────────────────────── */
.thinking-bubble {
    background: linear-gradient(135deg, #1c2333, #1a2332);
    border: 1px solid #30363d;
    color: #8b949e;
    padding: 14px 20px;
    border-radius: 18px 18px 18px 4px;
    font-size: 0.88rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

.dot-flashing {
    display: inline-flex;
    gap: 5px;
}

.dot-flashing span {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #58a6ff;
    animation: dotFlash 1.2s infinite ease-in-out;
}

.dot-flashing span:nth-child(2) { animation-delay: 0.2s; }
.dot-flashing span:nth-child(3) { animation-delay: 0.4s; }

@keyframes dotFlash {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
    40%            { opacity: 1;   transform: scale(1); }
}

/* ── Welcome card ────────────────────────────────────────────────────────── */
.welcome-card {
    background: linear-gradient(135deg, #1c2333, #1a2332);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    margin-top: 20px;
}

.welcome-card .icon { font-size: 3rem; margin-bottom: 16px; }

.welcome-card h3 {
    color: #e6edf3;
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 10px;
}

.welcome-card p {
    color: #8b949e;
    font-size: 0.9rem;
    max-width: 480px;
    margin: 0 auto 24px;
    line-height: 1.7;
}

.suggestion-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 20px;
}

.chip {
    background: #161b22;
    border: 1px solid #30363d;
    color: #8b949e;
    padding: 7px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s;
}

.chip:hover {
    border-color: #58a6ff;
    color: #58a6ff;
    background: rgba(88, 166, 255, 0.05);
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
.section-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 20px 0;
}

/* ── Scrollable chat area ────────────────────────────────────────────────── */
.chat-scroll-area {
    max-height: 62vh;
    overflow-y: auto;
    padding-right: 4px;
    scroll-behavior: smooth;
}

.chat-scroll-area::-webkit-scrollbar { width: 4px; }
.chat-scroll-area::-webkit-scrollbar-track { background: transparent; }
.chat-scroll-area::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 4px;
}

/* ── Input area ─────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 14px !important;
}

[data-testid="stChatInput"] textarea {
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Streamlit element tweaks ────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    font-family: 'Inter', sans-serif;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

.stFileUploader {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
}

/* ── Sidebar elements ────────────────────────────────────────────────────── */
.sidebar-section-title {
    color: #6e7681;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 20px 0 10px 0;
}

/* hide default streamlit header */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages":     [],          # [{"role": "user"|"assistant", "content": str, "sources": list}]
        "chat_history": [],          # [{"role":..., "content":...}] for backend
        "loaded_pdfs":  [],          # list of {"name": str, "path": str, "size": int}
        "indexed":      False,
        "indexing":     False,
        "index_error":  "",          # last indexing error message (persists across reruns)
        "session_id":   str(uuid.uuid4())[:8],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def save_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to disk, return its path."""
    dest = UPLOAD_DIR / f"{uploaded_file.name}"
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest)

def render_sources(sources: list[Source]):
    if not sources:
        return

    # Deduplicate by (doc_title, section_heading)
    seen = set()
    unique_sources = []

    for s in sources:
        key = (s.doc_title, s.section_heading)
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    st.markdown(
        '<div class="sources-header">📎 Sources used</div>',
        unsafe_allow_html=True
    )

    chips_html = ""

    for s in unique_sources:
        icon = "📊" if s.is_table else "📄"

        # Safe document title truncation
        doc_title = s.doc_title[:40]
        if len(s.doc_title) > 40:
            doc_title += "..."

        # Safe section heading truncation
        section_html = ""
        if s.section_heading and s.section_heading != "N/A":
            section_heading = s.section_heading[:35]
            if len(s.section_heading) > 35:
                section_heading += "..."

            section_html = f" · <em>{section_heading}</em>"

        chips_html += (
            f'<span class="source-chip">'
            f'{icon} '
            f'<span style="color:#e6edf3;font-weight:500">'
            f'{doc_title}'
            f'</span>'
            f'{section_html}'
            f'</span> '
        )

    st.markdown(chips_html, unsafe_allow_html=True)

# def render_sources(sources: list[Source]):
#     if not sources:
#         return
#     # Deduplicate by (doc_title, section_heading) — RAG_Pipeline returns one
#     # Source per chunk ID (up to 10), so the same section can appear many times.
#     seen = set()
#     unique_sources = []
#     for s in sources:
#         key = (s.doc_title, s.section_heading)
#         if key not in seen:
#             seen.add(key)
#             unique_sources.append(s)

#     st.markdown('<div class="sources-header">📎 Sources used</div>', unsafe_allow_html=True)
#     chips_html = ""
#     for s in unique_sources:
#         icon = "📊" if s.is_table else "📄"
#         chips_html += (
#             f'<span class="source-chip">'
#             f'{icon} '
#             f'<span style="color:#e6edf3;font-weight:500">'
#             f'{s.doc_title[:40]}{"…" if len(s.doc_title) > 40 else ""}'
#             f'</span>'
#             f'{"" if not s.section_heading or s.section_heading == "N/A" else f" · <em>{s.section_heading[:35]}{"…" if len(s.section_heading) > 35 else ""}</em>"}'
#             f'</span> '
#         )
#     st.markdown(chips_html, unsafe_allow_html=True)


def render_message(msg: dict):
    role    = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])

    if role == "user":
        st.markdown(
            f'<div class="msg-user">'
            f'  <div class="bubble-user">{content}</div>'
            f'  <div class="msg-avatar avatar-user">👤</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="msg-assistant">'
            f'  <div class="msg-avatar avatar-assistant">🔬</div>'
            f'  <div class="bubble-assistant">{content}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        if sources:
            st.markdown('<div style="padding-left:44px">', unsafe_allow_html=True)
            render_sources(sources)
            st.markdown('</div>', unsafe_allow_html=True)


def do_indexing():
    """Save and index uploaded PDFs via backend_interface.index_pdfs()."""
    paths = [p["path"] for p in st.session_state.loaded_pdfs]
    result = index_pdfs(paths)
    st.session_state.indexed      = result.success
    st.session_state.indexing     = False
    st.session_state.index_error  = "" if result.success else result.message
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / brand
    st.markdown("""
    <div style="padding: 8px 0 20px 0; border-bottom: 1px solid #21262d; margin-bottom: 20px;">
        <div style="font-size: 1.5rem; font-weight: 700; color: #e6edf3; display:flex; align-items:center; gap:10px;">
            🔬 <span style="background: linear-gradient(135deg, #60a5fa, #a78bfa);
                            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Research RAG
            </span>
        </div>
        <div style="color:#6e7681; font-size:0.78rem; margin-top:4px;">
            AI-Powered Research Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── System status ─────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">System Status</div>', unsafe_allow_html=True)
    if st.session_state.indexing:
        st.markdown('<span class="status-badge status-indexing">⚙ Indexing…</span>', unsafe_allow_html=True)
    elif st.session_state.indexed:
        n = len(st.session_state.loaded_pdfs)
        st.markdown(f'<span class="status-badge status-ready">✓ Ready · {n} PDF{"s" if n!=1 else ""} loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-idle">◌ No documents loaded</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">📂 Upload Research Papers</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        label_visibility="collapsed",
        help="Upload one or more PDF research papers to query.",
    )

    if uploaded:
        new_names = {f.name for f in uploaded}
        existing  = {p["name"] for p in st.session_state.loaded_pdfs}
        added_any = False

        for uf in uploaded:
            if uf.name not in existing:
                path = save_uploaded_file(uf)
                st.session_state.loaded_pdfs.append({
                    "name": uf.name,
                    "path": path,
                    "size": uf.size,
                })
                added_any = True

        if added_any:
            st.session_state.indexed = False   # need re-index

    # ── Loaded PDFs list ──────────────────────────────────────────────────────
    if st.session_state.loaded_pdfs:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">Loaded Documents</div>', unsafe_allow_html=True)

        for pdf in st.session_state.loaded_pdfs:
            st.markdown(
                f'<div class="pdf-pill">'
                f'  <span class="pdf-icon">📄</span>'
                f'  <span class="pdf-name" title="{pdf["name"]}">{pdf["name"]}</span>'
                f'  <span class="pdf-size">{fmt_bytes(pdf["size"])}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Index button
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(
                "⚡ Index Documents",
                key="btn_index",
                use_container_width=True,
                disabled=st.session_state.indexed or st.session_state.indexing,
            ):
                st.session_state.indexing = True
                st.session_state.index_error = ""  # clear previous error
                with st.spinner("Parsing & indexing…"):
                    result = do_indexing()
                if result.success:
                    st.session_state.index_error = ""
                st.rerun()

        # ── Show persistent indexing result (survives rerun) ──────────────────
        if st.session_state.indexed:
            st.success(f"✓ Indexed {len(st.session_state.loaded_pdfs)} document(s) — ready to query.")
        elif st.session_state.index_error:
            st.error(f"✗ Indexing failed")
            with st.expander("Show error details", expanded=False):
                st.code(st.session_state.index_error, language="python")

        with col2:
            if st.button("🗑", key="btn_clear", use_container_width=True, help="Remove all documents"):
                # Delete saved files
                for pdf in st.session_state.loaded_pdfs:
                    try:
                        os.remove(pdf["path"])
                    except OSError:
                        pass
                st.session_state.loaded_pdfs  = []
                st.session_state.indexed       = False
                st.session_state.messages      = []
                st.session_state.chat_history  = []
                st.rerun()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Settings expander ─────────────────────────────────────────────────────
    # with st.expander("⚙ Settings", expanded=False):
    #     st.session_state.top_k = st.slider(
    #         "Top-K children retrieved",
    #         min_value=2, max_value=10, value=5,
    #         help="Number of similar text chunks to retrieve from the vector store."
    #     )
    #     st.session_state.top_k_parents = st.slider(
    #         "Max parent sections",
    #         min_value=1, max_value=6, value=3,
    #         help="Number of unique paper sections surfaced as context."
    #     )
    #     st.session_state.show_sources = st.toggle(
    #         "Show source citations", value=True,
    #         help="Display which papers and sections were used to answer."
    #     )
    #     if st.button("🗑 Clear chat history", key="btn_clear_chat"):
    #         st.session_state.messages     = []
    #         st.session_state.chat_history = []
    #         st.rerun()

    # ── Session info ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-top:20px; color:#484f58; font-size:0.72rem; padding:8px 0;">
        Session · {st.session_state.session_id}<br>
        Model · nex-agi/nex-n2-pro<br>
        Embedding · BAAI/bge-base-en-v1.5
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 Research RAG Assistant</h1>
    <p>Ask questions about your research papers — get cited, context-aware answers powered by nex-agi/nex-n2-pro</p>
</div>
""", unsafe_allow_html=True)


# ── Suggestion chips (if no conversation yet) ─────────────────────────────────
SUGGESTIONS = [
    "What are the main contributions of these papers?",
    "What factors affect developer productivity the most?",
    "How does code quality causally impact productivity?",
    "Explain the attention mechanism in transformers.",
    "What are common threats to validity in these studies?",
    "Compare the methodologies used across the papers.",
]

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="icon">📚</div>
        <h3>Start your research conversation</h3>
        <p>Upload your research papers using the sidebar, then ask any question.
           The system will retrieve relevant context and generate a cited answer.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='text-align:center; margin-top:16px; color:#6e7681; font-size:0.82rem;'>Try a suggestion:</div>", unsafe_allow_html=True)

    cols = st.columns(3)
    for i, sug in enumerate(SUGGESTIONS):
        with cols[i % 3]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state._pending_question = sug
                st.rerun()


# ── Chat history ──────────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown('<div class="chat-scroll-area">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        render_message(msg)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input(
    placeholder="Ask a question about your research papers…",
    key="chat_input",
    disabled=st.session_state.indexing,
)

# Pick up suggestion click
if hasattr(st.session_state, "_pending_question") and st.session_state._pending_question:
    question = st.session_state._pending_question
    st.session_state._pending_question = None


# ─────────────────────────────────────────────────────────────────────────────
# HANDLE QUESTION
# ─────────────────────────────────────────────────────────────────────────────
if question and question.strip():

    # Gate: must have at least one PDF indexed
    if not st.session_state.loaded_pdfs:
        st.warning("⚠️ Please upload and index at least one PDF before asking questions.", icon="📄")
        st.stop()

    if not st.session_state.indexed:
        st.warning("⚠️ Documents are uploaded but not yet indexed. Click **⚡ Index Documents** in the sidebar.", icon="⚙️")
        st.stop()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": question})
    render_message({"role": "user", "content": question})

    # Show thinking animation
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown(
        '<div class="msg-assistant">'
        '  <div class="msg-avatar avatar-assistant">🔬</div>'
        '  <div class="thinking-bubble">Searching papers & generating answer'
        '    <div class="dot-flashing"><span></span><span></span><span></span></div>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Call backend ──────────────────────────────────────────────────────────
    pdf_paths = [p["path"] for p in st.session_state.loaded_pdfs]
    try:
        result: AskResult = ask(
            question     = question,
            pdf_paths    = pdf_paths,
            chat_history = st.session_state.chat_history,
        )
        answer  = result.answer  if not result.error else f"❌ Error: {result.error}"
        sources = result.sources if not result.error else []
    except Exception as exc:
        answer  = f"❌ Backend error: {exc}"
        sources = []

    # Clear thinking bubble
    thinking_placeholder.empty()

    # Update conversation history (for next turn)
    st.session_state.chat_history.append({"role": "user",      "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Append assistant message
    show_sources = getattr(st.session_state, "show_sources", True)
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources if show_sources else [],
    })

    # Render assistant message
    render_message(st.session_state.messages[-1])

    st.rerun()
