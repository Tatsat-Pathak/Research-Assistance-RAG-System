"""
app.py — Research RAG Assistant · Streamlit UI
=================================================
Modern, reactive Streamlit interface for Hugging Face Spaces & local execution.
"""
import os
import sys
import time
import uuid
import shutil
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import streamlit as st

# Ensure deployment directory is on sys.path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# Disable ChromaDB anonymous telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Direct imports from local RAG_Pipeline
import RAG_Pipeline
from RAG_Pipeline import (
    parse_document,
    parse_documents_batch,
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
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Source:
    """One retrieved source displayed under an assistant message."""
    doc_title:       str
    section_heading: str
    similarity:      float = 0.0
    is_table:        bool  = False
    snippet:         str   = ""

@dataclass
class AskResult:
    """Return type of ask()."""
    answer:    str
    reasoning: str = ""
    sources:   List[Source] = field(default_factory=list)
    error:     Optional[str] = None

@dataclass
class IndexResult:
    """Return type of index_pdfs()."""
    success:   bool
    doc_count: int = 0
    message:   str = ""


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETONS (Cached across Streamlit reruns)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_embedding_model():
    return load_embedded_model("BAAI/bge-base-en-v1.5")

@st.cache_resource(show_spinner=False)
def _get_chroma_client():
    return initialize_vector_database()


# ─────────────────────────────────────────────────────────────────────────────
# INDEXING LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def index_pdfs(pdf_paths: List[str]) -> IndexResult:
    try:
        # 1. Parse PDFs in parallel using multiprocessing
        papers_section_heading = parse_documents_batch(pdf_paths)
        parsed_count = len(papers_section_heading)

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
        RAG_Pipeline.embedding_model = model
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

        # 6. Store in ChromaDB with metadata
        add_kwargs = {
            "ids": all_ids,
            "documents": all_chunks,
            "embeddings": embeddings.tolist(),
        }
        if getattr(RAG_Pipeline, "last_metadatas", None):
            add_kwargs["metadatas"] = RAG_Pipeline.last_metadatas

        collection.add(**add_kwargs)
        RAG_Pipeline.collection = collection

        return IndexResult(
            success=True, doc_count=parsed_count,
            message=f"Indexed {parsed_count} document(s) → {len(all_chunks)} chunks stored.",
        )

    except Exception as exc:
        tb = traceback.format_exc()
        return IndexResult(success=False, doc_count=0,
                           message=f"Indexing error: {exc}\n\nTraceback:\n{tb}")


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL & QUERY LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def ask(question: str, top_k: int = 15, pdf_paths=None, chat_history=None) -> AskResult:
    try:
        # Ensure embedding model is loaded
        if not hasattr(RAG_Pipeline, "embedding_model") or RAG_Pipeline.embedding_model is None:
            RAG_Pipeline.embedding_model = _get_embedding_model()

        # Ensure collection is loaded
        if not hasattr(RAG_Pipeline, "collection") or RAG_Pipeline.collection is None:
            client = _get_chroma_client()
            try:
                RAG_Pipeline.collection = client.get_collection(RAG_Pipeline.COLLECTION_NAME)
            except Exception:
                pass

        if not hasattr(RAG_Pipeline, "collection") or RAG_Pipeline.collection is None:
            return AskResult(answer="",
                             error="No documents indexed yet. Please upload and index PDFs first.")

        # 1. Build prompt + retrieve context
        prompt, source_titles, chunk_ids = query_builder(question, top_k=top_k)

        # 2. Generate answer with OpenRouter LLM
        answer = answer_generator(prompt)

        if not answer.get('success'):
            return AskResult(answer="", error=f"LLM Error: {answer.get('error', 'Unknown problem')}")

        answer_content = answer.get('content', '') or ''
        reasoning_trace = answer.get('reasoning', '') or ''

        # 3. Build Source objects from metadata or chunk IDs
        sources: List[Source] = []
        seen_ids: set = set()

        id_to_meta = {}
        try:
            coll_data = RAG_Pipeline.collection.get(ids=chunk_ids, include=["metadatas"])
            if coll_data and coll_data.get("ids") and coll_data.get("metadatas"):
                for mid, mdata in zip(coll_data["ids"], coll_data["metadatas"]):
                    if mdata:
                        id_to_meta[mid] = mdata
        except Exception:
            pass

        for cid in chunk_ids:
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            meta = id_to_meta.get(cid)
            if meta and meta.get("paper_title"):
                doc_title_clean = meta["paper_title"]
                section_heading = meta.get("section_path") or meta.get("section", "N/A")
                is_table = meta.get("chunk_type") == "table"
            else:
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
                is_table = "table" in cid.lower()

            sources.append(Source(
                doc_title       = doc_title_clean if doc_title_clean else "Research Paper",
                section_heading = section_heading if section_heading else "N/A",
                similarity      = 0.0,
                is_table        = is_table,
                snippet         = "",
            ))

        return AskResult(answer=answer_content, reasoning=reasoning_trace, sources=sources)

    except Exception as exc:
        return AskResult(answer="", error=f"Pipeline error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research RAG Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_DIR = Path("uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e6edf3;
}

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

.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 16px;
}

.msg-assistant {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 16px;
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
}

.pdf-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #1c2333;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
}

.pdf-pill .pdf-name {
    color: #e6edf3;
    font-size: 0.83rem;
    font-weight: 500;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

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

.sidebar-section-title {
    color: #6e7681;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 20px 0 10px 0;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages":     [],          # [{"role": "user"|"assistant", "content": str, "reasoning": str, "sources": list}]
        "chat_history": [],
        "loaded_pdfs":  [],          # [{"name": str, "path": str, "size": int}]
        "indexed":      False,
        "indexing":     False,
        "index_error":  "",
        "session_id":   str(uuid.uuid4())[:8],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────────
# UI RENDERING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def save_uploaded_file(uploaded_file) -> str:
    dest = UPLOAD_DIR / f"{uploaded_file.name}"
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest)


def render_sources(sources: List[Source]):
    if not sources:
        return

    seen = set()
    unique_sources = []
    for s in sources:
        key = (s.doc_title, s.section_heading)
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    st.markdown('<div class="sources-header">📎 Sources Used</div>', unsafe_allow_html=True)
    chips_html = ""
    for s in unique_sources:
        icon = "📊" if s.is_table else "📄"
        doc_title = s.doc_title[:40] + ("..." if len(s.doc_title) > 40 else "")
        section_html = ""
        if s.section_heading and s.section_heading != "N/A":
            sec_text = s.section_heading[:35] + ("..." if len(s.section_heading) > 35 else "")
            section_html = f" · <em>{sec_text}</em>"

        chips_html += (
            f'<span class="source-chip">'
            f'{icon} <span style="color:#e6edf3;font-weight:500">{doc_title}</span>'
            f'{section_html}'
            f'</span> '
        )
    st.markdown(chips_html, unsafe_allow_html=True)


def render_message(msg: dict):
    role = msg["role"]
    content = msg["content"]
    reasoning = msg.get("reasoning", "")
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
        if reasoning:
            with st.expander("🧠 View Deep Reasoning Trace", expanded=False):
                st.markdown(reasoning)

        if sources:
            st.markdown('<div style="padding-left:44px">', unsafe_allow_html=True)
            render_sources(sources)
            st.markdown('</div>', unsafe_allow_html=True)


def do_indexing():
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

    # System status
    st.markdown('<div class="sidebar-section-title">System Status</div>', unsafe_allow_html=True)
    if st.session_state.indexing:
        st.markdown('<span class="status-badge status-indexing">⚙ Indexing…</span>', unsafe_allow_html=True)
    elif st.session_state.indexed:
        n = len(st.session_state.loaded_pdfs)
        st.markdown(f'<span class="status-badge status-ready">✓ Ready · {n} PDF{"s" if n!=1 else ""} loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-idle">◌ No documents loaded</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload
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
        existing = {p["name"] for p in st.session_state.loaded_pdfs}
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
            st.session_state.indexed = False

    # Loaded PDFs list
    if st.session_state.loaded_pdfs:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">Loaded Documents</div>', unsafe_allow_html=True)

        for pdf in st.session_state.loaded_pdfs:
            st.markdown(
                f'<div class="pdf-pill">'
                f'  <span>📄</span>'
                f'  <span class="pdf-name" title="{pdf["name"]}">{pdf["name"]}</span>'
                f'  <span style="color:#6e7681; font-size:0.75rem;">{fmt_bytes(pdf["size"])}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(
                "⚡ Index Documents",
                key="btn_index",
                use_container_width=True,
                disabled=st.session_state.indexed or st.session_state.indexing,
            ):
                st.session_state.indexing = True
                st.session_state.index_error = ""
                with st.spinner("Parsing & indexing…"):
                    result = do_indexing()
                st.rerun()

        with col2:
            if st.button("🗑", key="btn_clear", use_container_width=True, help="Remove all documents"):
                for pdf in st.session_state.loaded_pdfs:
                    try:
                        os.remove(pdf["path"])
                    except OSError:
                        pass
                st.session_state.loaded_pdfs  = []
                st.session_state.indexed      = False
                st.session_state.messages     = []
                st.session_state.chat_history = []
                st.rerun()

        if st.session_state.indexed:
            st.success(f"✓ Indexed {len(st.session_state.loaded_pdfs)} document(s) — ready to query.")
        elif st.session_state.index_error:
            st.error("✗ Indexing failed")
            with st.expander("Show error details", expanded=False):
                st.code(st.session_state.index_error, language="python")

    st.markdown("<hr style='border:none; border-top:1px solid #21262d; margin:20px 0;'>", unsafe_allow_html=True)

    # Settings
    with st.expander("⚙️ Retrieval & API Settings", expanded=False):
        st.session_state.top_k = st.slider(
            "Context Chunks (Top-K)",
            min_value=3, max_value=30, value=st.session_state.get("top_k", 15),
            help="Number of most relevant text chunks retrieved from vector store to answer the question."
        )
        st.session_state.show_sources = st.toggle(
            "Show source citations", value=st.session_state.get("show_sources", True),
            help="Display which papers and sections were used to answer."
        )
        if st.button("🧹 Clear Chat History", key="btn_clear_chat", use_container_width=True):
            st.session_state.messages     = []
            st.session_state.chat_history = []
            st.rerun()

    st.markdown(f"""
    <div style="margin-top:20px; color:#484f58; font-size:0.72rem; padding:8px 0;">
        Session · {st.session_state.session_id}<br>
        Backend · OpenRouter Nemotron / DeepSeek<br>
        Embedding · BAAI/bge-base-en-v1.5
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 Research RAG Assistant</h1>
    <p>Ask in-depth questions about your research papers — get cited, context-aware answers with deep reasoning traces.</p>
</div>
""", unsafe_allow_html=True)

SUGGESTIONS = [
    "What are the main contributions of these papers?",
    "What are common threats to validity in these studies?",
    "Compare the methodologies and benchmarks used across the papers.",
]

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="icon">📚</div>
        <h3>Start Your Research Conversation</h3>
        <p>Upload your research papers using the sidebar, then ask any question.
           The assistant will retrieve relevant context, citations, and generate a synthesized answer.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='text-align:center; margin-top:16px; color:#6e7681; font-size:0.82rem;'>Suggested research questions:</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, sug in enumerate(SUGGESTIONS):
        with cols[i % 3]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state._pending_question = sug
                st.rerun()

# Render chat history
if st.session_state.messages:
    for msg in st.session_state.messages:
        render_message(msg)

# Chat input
question = st.chat_input(
    placeholder="Ask a question about your research papers…",
    key="chat_input",
    disabled=st.session_state.indexing,
)

if hasattr(st.session_state, "_pending_question") and st.session_state._pending_question:
    question = st.session_state._pending_question
    st.session_state._pending_question = None

if question and question.strip():
    if not st.session_state.loaded_pdfs:
        st.warning("⚠️ Please upload and index at least one PDF in the sidebar before asking questions.", icon="📄")
        st.stop()

    if not st.session_state.indexed:
        st.warning("⚠️ Documents are uploaded but not yet indexed. Click **⚡ Index Documents** in the sidebar.", icon="⚙️")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    render_message({"role": "user", "content": question})

    with st.spinner("Searching papers & generating answer…"):
        pdf_paths = [p["path"] for p in st.session_state.loaded_pdfs]
        top_k_val = st.session_state.get("top_k", 15)
        result: AskResult = ask(
            question     = question,
            top_k        = top_k_val,
            pdf_paths    = pdf_paths,
            chat_history = st.session_state.chat_history,
        )

    answer    = result.answer if not result.error else f"❌ Error: {result.error}"
    reasoning = result.reasoning if not result.error else ""
    sources   = result.sources if not result.error else []

    st.session_state.chat_history.append({"role": "user",      "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    show_sources = getattr(st.session_state, "show_sources", True)
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   answer,
        "reasoning": reasoning,
        "sources":   sources if show_sources else [],
    })

    st.rerun()
