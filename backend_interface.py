"""
backend_interface.py
--------------------
Wires the Streamlit UI to the user's RAG_Pipeline.py functions.

The Streamlit app (app.py) imports and calls:
  • index_pdfs(pdf_paths)                   ->  IndexResult
  • ask(question, pdf_paths, chat_history)  ->  AskResult
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Return-type dataclasses (UI contract — do not change field names)
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
# Import user's pipeline — RAG_Pipeline.py is NOT modified
# ─────────────────────────────────────────────────────────────────────────────

from RAG_Pipeline import (
    parse_document,
    chunks_generation,
    combining_chunks,
    load_embedded_model,
    create_embedding_text,
    initialize_vector_database,
    initialize_collections,
    get_collection,
    add_in_collection,
    query_builder,
    answer_generator,
)
import RAG_Pipeline  # we set module-level globals (embedding_model, collection) here


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons  (loaded once, reused across requests)
# ─────────────────────────────────────────────────────────────────────────────

_embedding_model = None   # loaded lazily on first call
_chroma_client   = None   # PersistentClient instance


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
# index_pdfs
# ─────────────────────────────────────────────────────────────────────────────

def index_pdfs(pdf_paths: List[str]) -> IndexResult:
    """
    Parse, chunk, embed and store the uploaded PDFs in ChromaDB.

    Workflow (all via RAG_Pipeline functions):
      1. parse_document()        — pymupdf4llm -> clean -> section extraction
      2. paper_triming()         — called internally by parse_document
      3. chunks_generation()     — 500-char overlapping chunks per section
      4. combining_chunks()      — flatten into one list
      5. load_embedded_model()   — BAAI/bge-base-en-v1.5 (singleton)
      6. create_embedding_text() — batch encode all chunks
      7. initialize_vector_database() + initialize_collections() — ChromaDB
      8. add_in_collection()     — upsert embeddings + docs
    """
    try:
        # ── 1. Parse every PDF ────────────────────────────────────────────────
        papers_section_heading: Dict[str, list] = {}
        parsed_count = 0

        for path in pdf_paths:
            file_name, _, sections = parse_document(path)
            if sections:
                papers_section_heading[file_name] = sections
                parsed_count += 1

        if not papers_section_heading:
            return IndexResult(
                success=False,
                doc_count=0,
                message="No readable sections found in the uploaded PDFs.",
            )

        # ── 2. Generate chunks ────────────────────────────────────────────────
        paper_chunks = chunks_generation(papers_section_heading, chunk_size=500)

        # ── 3. Flatten ────────────────────────────────────────────────────────
        all_chunks, all_ids = combining_chunks(paper_chunks)

        if not all_chunks:
            return IndexResult(
                success=False,
                doc_count=parsed_count,
                message="Documents parsed but no chunks were generated.",
            )

        # ── 4. Embed ──────────────────────────────────────────────────────────
        model = _get_embedding_model()
        # Inject into RAG_Pipeline module so query_builder() can use it
        RAG_Pipeline.embedding_model = model

        embeddings, shape = create_embedding_text(model, all_chunks, size=64)

        # ── 5. ChromaDB — drop & recreate collection for a fresh index ────────
        # Delete ALL existing collections regardless of name, so we always
        # start clean (handles collection renames in RAG_Pipeline.py too).
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

        # ── 6. Store ──────────────────────────────────────────────────────────
        # NOTE: add_in_collection() in RAG_Pipeline.py has a typo:
        #   `embedding=` instead of `embeddings=` (ChromaDB keyword).
        # Calling collection.add() directly with the correct argument name.
        collection.add(
            ids        = all_ids,
            documents  = all_chunks,
            embeddings = embeddings.tolist(),
        )

        # Inject collection into RAG_Pipeline so query_builder() can query it
        RAG_Pipeline.collection = collection

        return IndexResult(
            success=True,
            doc_count=parsed_count,
            message=(
                f"Indexed {parsed_count} document(s) → "
                f"{len(all_chunks)} chunks stored in ChromaDB."
            ),
        )

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        return IndexResult(success=False, doc_count=0, message=f"Indexing error: {exc}\n\nTraceback:\n{tb}")


# ─────────────────────────────────────────────────────────────────────────────
# ask
# ─────────────────────────────────────────────────────────────────────────────

def ask(
    question:     str,
    pdf_paths:    Optional[List[str]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> AskResult:
    """
    Run the full RAG query pipeline and return the LLM answer + sources.

    Workflow:
      1. query_builder()    — embeds query, searches ChromaDB, builds prompt
      2. answer_generator() — calls OpenRouter LLM, returns cleaned answer text
      3. Build Source objects from the retrieved chunk IDs
    """
    try:
        # Guard: embedding_model and collection must be set (happens after index)
        if not hasattr(RAG_Pipeline, "collection") or RAG_Pipeline.collection is None:
            return AskResult(
                answer="",
                error="No documents indexed yet. Please upload and index PDFs first.",
            )

        # ── 1. Build prompt + retrieve context ────────────────────────────────
        prompt, source_titles, chunk_ids = query_builder(question)

        # ── 2. Generate answer ────────────────────────────────────────────────
        answer = answer_generator(prompt)

        if answer == "Error":
            return AskResult(
                answer="",
                error="LLM API returned an error. Check your OPEN_ROUTER_API_KEY in .env.",
            )

        # ── 3. Build Source objects from chunk IDs ────────────────────────────
        #
        # Chunk ID format (from RAG_Pipeline.chunks_generation):
        #   "{file_name}_{heading}_{section_index}_chunk_{chunk_index}"
        #   e.g. "01_paper.pdf_Introduction_0_chunk_2"
        #
        # We parse:  doc_title  = everything before the last .pdf occurrence
        #            section    = part between file name and _chunk_
        sources: List[Source] = []
        seen_ids: set = set()

        for cid in chunk_ids:
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            # Extract doc title (up to and including ".pdf")
            pdf_marker = cid.find(".pdf")
            if pdf_marker != -1:
                doc_title = cid[: pdf_marker + 4]   # includes ".pdf"
                remainder = cid[pdf_marker + 5:]    # skip ".pdf_"
            else:
                doc_title = cid
                remainder = ""

            # Extract section heading (between doc_title and "_N_chunk_N")
            # Pattern: {heading}_{section_index}_chunk_{chunk_index}
            chunk_marker = remainder.rfind("_chunk_")
            if chunk_marker != -1:
                before_chunk = remainder[:chunk_marker]
                # Remove trailing section index "_N"
                last_underscore = before_chunk.rfind("_")
                section_heading = before_chunk[:last_underscore] if last_underscore != -1 else before_chunk
            else:
                section_heading = remainder

            # Clean up heading (replace underscores from ID construction)
            section_heading = section_heading.replace("_", " ").strip()
            doc_title_clean = doc_title.replace("_", " ").replace(".pdf", "").strip()

            sources.append(Source(
                doc_title       = doc_title_clean if doc_title_clean else doc_title,
                section_heading = section_heading if section_heading else "N/A",
                similarity      = 0.0,   # RAG_Pipeline doesn't return distances
                is_table        = False,
                snippet         = "",
            ))

        return AskResult(answer=answer, sources=sources)

    except Exception as exc:
        return AskResult(answer="", error=f"Pipeline error: {exc}")
