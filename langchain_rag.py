"""
langchain_rag.py — a small LangChain version of the Research Assistance RAG System.

What this is, and what it is not
---------------------------------
* It is a *second wiring* of the same RAG pipeline, not a rewrite and not a
  replacement for RAG_Pipeline.py / app.py. The from-scratch pipeline stays
  the main app.
* It reuses the parts that already work and are already tested:
  MinerU parsing (``parse_documents_batch``), section-aware chunking
  (``chunks_generation`` / ``chunk_papers``, ~300 words), the
  BAAI/bge-base-en-v1.5 embeddings, and the same persistent Chroma
  collection (``./upload_pdfs_chroma_db``, collection
  ``chroma_db_documents``). A paper indexed by the main app can be
  searched by this file, and vice versa — the chunk ids, documents
  (``embed_text``) and metadata are identical.
* Only the retrieval + LLM wiring is LangChain:
  ``langchain_chroma.Chroma`` as the vector store,
  ``langchain_huggingface.HuggingFaceEmbeddings`` for embeddings,
  ``langchain_openai.ChatOpenAI`` pointed at OpenRouter for the LLM, and
  a small LCEL chain (prompt | LLM | output parser).
* It keeps the one fix that matters most in this repo: per-paper
  retrieval. LangChain's built-in retriever does one global top-k
  search, which is exactly the bug PR #2 fixed — with 5 papers, the
  2–3 best-matching papers filled every slot and the rest never reached
  the LLM. So retrieval here is done per paper on purpose (top_k split
  evenly across papers, each paper's Abstract always included), the
  same as ``query_builder()`` in RAG_Pipeline.py.
* There is no LangGraph in this file, on purpose. LangGraph is for
  multi-step agents (loops, tools, state graphs). A question-answering
  RAG chain is a straight line — retrieve, prompt, answer — and
  LangChain alone is the right tool for it. Learning LangGraph next
  would mean building an agent on top of this chain, not replacing it.

Install (add to requirements.txt, or install in the same venv as the app):
    langchain>=1.5,<2
    langchain-chroma>=1.1,<2
    langchain-huggingface>=1.2,<2
    langchain-openai>=1.8,<2
    (tested 2026-11-16: langchain 1.5.0, langchain-chroma 1.1.0,
    langchain-huggingface 1.2.3, langchain-openai 1.8.0, chromadb 1.5.9)

Run on your laptop (the app's normal environment — MinerU + bge need
real CPU/GPU and the model downloads, so this is not a VM/CI job):
    # 1. index PDFs (reuses the main app's parse cache + Chroma store)
    python langchain_rag.py --pdfs paper1.pdf paper2.pdf
    # 2. ask a question (needs OPENROUTER_API_KEY in .env, same as app.py)
    python langchain_rag.py --ask "what does each paper talk about?" --top-k 15

    # Skip indexing if the papers are already indexed by the Streamlit app:
    python langchain_rag.py --ask "what does each paper talk about?"

The OpenRouter key comes only from a local .env file, never from a UI
input — same rule as the main app.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Same constants as RAG_Pipeline.py — the two files share one store.
# They are duplicated here (not imported) so that importing this module
# does not pull in torch / MinerU / sentence-transformers. Those heavy
# imports happen lazily inside index_papers(), only when PDFs are parsed.
CHROMA_PATH = "./upload_pdfs_chroma_db"
COLLECTION_NAME = "chroma_db_documents"

DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_LLM_MODEL = "nvidia/nemotron-3.5-super-vl:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

PROMPT_TEMPLATE = """You are an expert research assistant helping researchers and students.
Give a precise and clean answer to the user question, clearly and accurately, using only the context provided.
If the answer is not in the context, say sorry, you can't answer it based on the given context.

IMPORTANT: the Context below is grouped by paper, one group per paper.
Cover EVERY paper in the Context, one section per paper, in the same
order as they appear. Do not skip a paper. If a paper's chunks do not
answer the question, say so for that paper instead of skipping it.

User question:
{question}

Context:
{context}
"""


# ---------------------------------------------------------------------------
# Setup: embeddings, vector store, LLM
# ---------------------------------------------------------------------------
def get_embeddings(model_name: str = DEFAULT_EMBED_MODEL):
    """BGE embeddings, normalised — the same model and settings as the main app."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(
    persist_directory: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    embeddings=None,
):
    """Open the shared Chroma collection through LangChain.

    If ``embeddings`` is None, the BGE model is loaded. Pass a fake
    embedding function in tests to avoid downloading the model.
    """
    if embeddings is None:
        embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def get_llm(model: str = DEFAULT_LLM_MODEL, api_key: Optional[str] = None, temperature: float = 0.0):
    """ChatOpenAI pointed at OpenRouter — same provider and default model as the main app."""
    load_dotenv()
    api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No OpenRouter API key set. Set OPENROUTER_API_KEY in a .env file, then ask again."
        )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Indexing — reuses the main app's parsing + chunking, then stores via LangChain
# ---------------------------------------------------------------------------
def _chunk_to_document(chunk: dict) -> Document:
    """One chunk dict from chunk_papers() -> one LangChain Document.

    page_content is the chunk's embed_text (the same text the main app
    stores in Chroma), so the two stores stay interchangeable. Metadata
    is the same set combining_chunks() produces.
    """
    return Document(
        page_content=chunk["embed_text"],
        metadata={
            "paper_title": chunk["paper_title"],
            "paper_id": chunk.get("paper_id", "") or "",
            "section": chunk["section"],
            "section_path": " > ".join(chunk["section_path"])
            if isinstance(chunk["section_path"], list)
            else str(chunk["section_path"]),
            "chunk_type": chunk.get("chunk_type", "body"),
            "label": chunk.get("label", "") or "",
            # Theme is kept when the chunk carries one (the literature-review
            # benchmark tags chunks by theme); absent for normal app chunks.
            "theme": chunk.get("theme", "") or "",
        },
    )


def index_chunks(chunks: list, vectorstore) -> int:
    """Store chunk dicts in the LangChain vector store. Returns how many were added.

    Chunks already in the collection (same chunk_id) are skipped, so
    re-running indexing — or indexing papers the Streamlit app already
    indexed — does not duplicate them.
    """
    if not chunks:
        return 0
    try:
        existing = set(vectorstore.get(include=[])["ids"] or [])
    except Exception:
        existing = set()
    fresh = [c for c in chunks if c["chunk_id"] not in existing]
    if not fresh:
        print(f"[LangChain] {len(chunks)} chunk(s) already indexed, nothing to add.")
        return 0
    docs = [_chunk_to_document(c) for c in fresh]
    ids = [c["chunk_id"] for c in fresh]
    vectorstore.add_documents(docs, ids=ids)
    print(f"[LangChain] Indexed {len(fresh)} new chunk(s) ({len(chunks) - len(fresh)} already present).")
    return len(fresh)


def index_papers(pdf_paths: List[str], chunk_size: int = 300, vectorstore=None, embeddings=None) -> int:
    """Parse PDFs and index them. Returns the number of chunks added.

    Parsing and chunking are the main app's functions — MinerU +
    section-aware ~300-word chunks — so a LangChain-indexed paper is
    chunked exactly like a main-app-indexed one.
    """
    if vectorstore is None:
        vectorstore = get_vectorstore(embeddings=embeddings)
    # Heavy imports, only when parsing is actually needed.
    from RAG_Pipeline import chunks_generation, parse_documents_batch

    sections = parse_documents_batch(list(pdf_paths))
    if not sections:
        print("[LangChain] No papers parsed, nothing indexed.")
        return 0
    chunks = chunks_generation(sections, chunk_size=chunk_size)
    return index_chunks(chunks, vectorstore)


# ---------------------------------------------------------------------------
# Retrieval — per paper, on purpose (see module docstring)
# ---------------------------------------------------------------------------
def list_paper_titles(vectorstore) -> List[str]:
    """Distinct paper titles in the collection, in first-seen order."""
    try:
        data = vectorstore.get(include=["metadatas"])
    except Exception:
        return []
    titles: List[str] = []
    for meta in data.get("metadatas") or []:
        title = (meta or {}).get("paper_title")
        if title and title not in titles:
            titles.append(title)
    return titles


def _abstract_doc_for_paper(vectorstore, paper_title: str) -> Optional[Document]:
    """The first Abstract chunk for a paper, or None."""
    try:
        # Chroma 1.5+ rejects a where/filter dict with two plain keys —
        # multiple conditions need $and. (The old two-key form raised
        # inside this try/except, so the Abstract was silently never
        # added and retrieve_per_paper returned exactly 2 chunks/paper.)
        docs = vectorstore.similarity_search(
            "abstract summary of the paper",
            k=1,
            filter={"$and": [{"paper_title": paper_title}, {"chunk_type": "abstract"}]},
        )
    except Exception:
        return None
    return docs[0] if docs else None


def retrieve_per_paper(vectorstore, query: str, top_k: int = 15) -> Dict[str, List[Document]]:
    """Retrieve chunks per paper: top_k split evenly, Abstract always included.

    Returns {paper_title: [Document, ...]}. With one paper (or if the
    title listing fails) this is a single global top-k search, exactly
    like query_builder() in RAG_Pipeline.py.
    """
    titles = list_paper_titles(vectorstore)
    per_paper: Dict[str, List[Document]] = {}

    if len(titles) <= 1:
        docs = vectorstore.similarity_search(query, k=top_k)
        for doc in docs:
            title = (doc.metadata or {}).get("paper_title") or "Unknown paper"
            per_paper.setdefault(title, []).append(doc)
        return per_paper

    per_paper_k = max(2, math.ceil(top_k / len(titles)))
    print(f"[LangChain] {len(titles)} papers indexed. Fetching top {per_paper_k} chunks per paper (budget top_k={top_k}).")
    for title in titles:
        docs = vectorstore.similarity_search(query, k=per_paper_k, filter={"paper_title": title})
        if not any((d.metadata or {}).get("chunk_type") == "abstract" for d in docs):
            abstract = _abstract_doc_for_paper(vectorstore, title)
            if abstract is not None and abstract.page_content not in {d.page_content for d in docs}:
                docs.insert(0, abstract)
        if docs:
            per_paper[title] = docs
    return per_paper


def build_context(per_paper: Dict[str, List[Document]]) -> str:
    """Group retrieved chunks by paper, in the same format as query_builder()."""
    context = ""
    for paper_title, docs in per_paper.items():
        context += f"\n===== Paper: {paper_title} =====\n"
        for doc in docs:
            meta = doc.metadata or {}
            section_path = meta.get("section_path") or meta.get("section", "Section")
            chunk_type = meta.get("chunk_type", "body")
            label = meta.get("label", "")
            type_header = (
                f" [{label}]"
                if label
                else (f" [{chunk_type}]" if chunk_type in ("table", "code", "figure_caption") else "")
            )
            context += f"\nSection : {section_path}{type_header}\nContent : {doc.page_content}\n\n-------"
    return context


# ---------------------------------------------------------------------------
# Chain — prompt | LLM | parser, with per-paper retrieval inside
# ---------------------------------------------------------------------------
def build_chain(llm=None, vectorstore=None, top_k: int = 15):
    """Return an LCEL Runnable: {"question": str} -> {"answer", "sources", "context"}.

    Retrieval lives inside the Runnable (via RunnableLambda) because
    LangChain's built-in retrievers do one global top-k search and
    would lose the per-paper guarantee.
    """
    if llm is None:
        llm = get_llm()
    if vectorstore is None:
        vectorstore = get_vectorstore()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    def _ask(inputs: dict) -> dict:
        question = inputs["question"]
        per_paper = retrieve_per_paper(vectorstore, question, top_k=top_k)
        sources = list(per_paper.keys())
        context = build_context(per_paper)
        print(f"[LangChain] Returning chunks covering {len(sources)} paper(s): {sources}")
        answer = chain.invoke({"question": question, "context": context})
        return {"answer": answer, "sources": sources, "context": context}

    return RunnableLambda(_ask)


def ask(question: str, top_k: int = 15, llm=None, vectorstore=None) -> dict:
    """Ask one question through the LangChain chain. Returns answer + sources."""
    return build_chain(llm=llm, vectorstore=vectorstore, top_k=top_k).invoke({"question": question})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Small LangChain version of the RAG system.")
    parser.add_argument("--pdfs", nargs="*", default=[], help="PDF files to index first")
    parser.add_argument("--ask", default=None, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=15, help="Total retrieval budget (split per paper)")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="OpenRouter model")
    parser.add_argument("--persist-dir", default=CHROMA_PATH, help="Chroma persist directory")
    args = parser.parse_args()

    load_dotenv()
    vectorstore = get_vectorstore(persist_directory=args.persist_dir)

    if args.pdfs:
        index_papers(args.pdfs, vectorstore=vectorstore)

    if args.ask:
        llm = get_llm(model=args.model)
        result = ask(args.ask, top_k=args.top_k, llm=llm, vectorstore=vectorstore)
        print("\n--- Answer ---\n")
        print(result["answer"])
        print("\n--- Sources ---")
        for title in result["sources"]:
            print(f" - {title}")
    elif not args.pdfs:
        parser.print_help()


if __name__ == "__main__":
    main()
