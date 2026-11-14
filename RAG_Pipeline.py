import os
import re
import json
import hashlib
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import chromadb
import torch
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    from mineru.parser.mineru_parser import MinerUParser
except ImportError:
    MinerUParser = None

from extract_papers import extract_paper, blocks_from_structured
from chunks_generated import chunk_papers

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration & Hardware / GPU Detection
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./upload_pdfs_chroma_db"
COLLECTION_NAME = "chroma_db_documents"
PARSE_CACHE_DIR = Path(os.getenv("RAG_PARSE_CACHE_DIR", "./parse_cache"))
PARSE_CACHE_DIR.mkdir(exist_ok=True)

def get_device() -> str:
    """Return 'cuda' if NVIDIA GPU is available and functional, else 'cpu'."""
    try:
        if torch.cuda.is_available():
            # Test a basic operation to verify sm architecture compatibility
            t = torch.zeros(1, device="cuda")
            _ = t + 1
            return "cuda"
    except Exception as e:
        logger.warning(f"CUDA detected but not functional with current PyTorch ({e}). Falling back to CPU.")
    return "cpu"

# Configure MinerU table engine device preference
os.environ.setdefault("MINERU_TABLE_DEVICE", "auto")

# Injected at runtime by api.py / app.py
embedding_model = None
collection = None
last_metadatas = None

_mineru_parser: Optional[MinerUParser] = None


def get_mineru_parser(tier: str = "basic"):
    """Singleton helper to initialize MinerUParser once per process."""
    global _mineru_parser
    if _mineru_parser is None:
        if MinerUParser is not None:
            _mineru_parser = MinerUParser(
                tier=tier,
                parse_mode="auto",
                image_analysis=False
            )
        else:
            raise RuntimeError("MinerU is not installed in this environment. Pre-indexed collections remain fully searchable.")
    return _mineru_parser


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for cache invalidation."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Document Parsing & Extraction (MinerU + extract_papers)
# ─────────────────────────────────────────────────────────────────────────────
def parse_document(file_path: str, parser: Optional[MinerUParser] = None) -> Tuple[str, str, Dict[str, list]]:
    """
    Parse a PDF file using MinerU with disk-caching support.
    
    Returns:
        (file_name, document_title, sections_dict)
    """
    file_name = os.path.basename(file_path)
    fallback_title = os.path.splitext(file_name)[0]

    # Check parse cache
    file_hash = compute_file_hash(file_path)
    cache_file = PARSE_CACHE_DIR / f"{file_hash}_mineru.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            title = cached.get("title", fallback_title)
            sections = cached.get("sections", {})
            return file_name, title, sections
        except Exception as e:
            logger.warning(f"Failed reading cache for {file_name}: {e}. Reparsing...")

    # Run MinerU parser with fallback
    if parser is None:
        try:
            parser = get_mineru_parser()
        except Exception:
            parser = None

    if parser is not None:
        doc = parser.parse(file_path)
        blocks = blocks_from_structured(doc.structured_content())
        title, sections, extras = extract_paper(blocks, fallback_title=fallback_title)
    else:
        # Fallback parser using pymupdf4llm / pypdf
        try:
            import pymupdf4llm
            md_text = pymupdf4llm.to_markdown(file_path, header=False, footer=False)
            title = fallback_title
            sections = {"Main": [{"type": "text", "content": md_text}]}
            extras = {}
        except Exception as e:
            logger.error(f"Fallback parse failed for {file_name}: {e}")
            title = fallback_title
            sections = {}
            extras = {}

    # Save to disk cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "title": title,
                "sections": sections,
                "extras": extras
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed writing cache for {file_name}: {e}")

    return file_name, title, sections


def _parse_single_doc_worker(file_path: str) -> Tuple[str, str, Dict[str, list]]:
    """Top-level worker function for ProcessPoolExecutor."""
    return parse_document(file_path)


def parse_documents_batch(file_paths: List[str], max_workers: Optional[int] = None) -> Dict[str, list]:
    """
    Parses multiple PDF files in parallel using ProcessPoolExecutor with spawn context.
    
    Returns:
        Dictionary of {document_title_or_filename: sections_dict}
    """
    if not file_paths:
        return {}

    # Single-file shortcut (no process pool overhead)
    if len(file_paths) == 1:
        file_name, title, sections = parse_document(file_paths[0])
        key = title if title and title != "untitled" else file_name
        return {key: sections} if sections else {}

    if max_workers is None:
        # Default is sequential (1 worker). Each worker process loads its own
        # copy of the MinerU parser + models, so 4 workers can use ~4x the RAM
        # and OOM a small host (Streamlit Cloud caps at ~2.7 GB). That was the
        # repeated crash. Raise it only on a host with RAM to spare, via
        # RAG_MAX_WORKERS=2, and never above 2.
        try:
            max_workers = int(os.getenv("RAG_MAX_WORKERS", "1"))
        except ValueError:
            max_workers = 1
        max_workers = max(1, min(max_workers, 2, len(file_paths)))

    if max_workers == 1:
        results: Dict[str, list] = {}
        for i, p in enumerate(file_paths, 1):
            try:
                file_name, title, sections = parse_document(p)
                if sections:
                    key = title if title and title != "untitled" else file_name
                    results[key] = sections
                    print(f"[{i}/{len(file_paths)}] Parsed '{file_name}' ({len(sections)} sections)")
            except Exception as exc:
                logger.error(f"Failed to parse {p}: {exc}")
        return results

    results: Dict[str, list] = {}
    ctx = multiprocessing.get_context("spawn")
    completed = 0
    total = len(file_paths)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        future_to_path = {executor.submit(_parse_single_doc_worker, p): p for p in file_paths}
        for future in as_completed(future_to_path):
            p = future_to_path[future]
            completed += 1
            try:
                file_name, title, sections = future.result()
                if sections:
                    key = title if title and title != "untitled" else file_name
                    results[key] = sections
                    print(f"[{completed}/{total}] Parsed '{file_name}' ({len(sections)} sections)")
            except Exception as exc:
                logger.error(f"Failed to parse {p} in worker: {exc}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Chunking (chunks_generated)
# ─────────────────────────────────────────────────────────────────────────────
def chunks_generation(papers_section_heading: dict, chunk_size: int = 300) -> list:
    """
    Generate hierarchical semantic chunks with section paths, equation support,
    table row splitting, and sentence-overlap packing.

    chunk_size is the target size of a prose chunk, in words. It is passed
    through to chunk_papers() — previously it was accepted here and silently
    ignored (chunking always used its own hardcoded 300-word target).

    Accepts:
        {paper_title_or_filename: sections_dict}
    Returns:
        List of chunk dicts
    """
    return chunk_papers(papers_section_heading, target_words=chunk_size)


def combining_chunks(paper_chunks) -> Tuple[list, list]:
    """
    Flatten and prepare chunks for embedding and ChromaDB indexing.
    
    Supports both:
      - Upgraded list of chunk dictionaries from chunk_papers
      - Legacy dict {title: {'chunks': ..., 'ids': ...}}
    """
    global last_metadatas

    if isinstance(paper_chunks, list):
        all_chunks = [c["embed_text"] for c in paper_chunks]
        all_ids = [c["chunk_id"] for c in paper_chunks]
        last_metadatas = [
            {
                "paper_title": c["paper_title"],
                "section": c["section"],
                "section_path": " > ".join(c["section_path"]) if isinstance(c["section_path"], list) else str(c["section_path"]),
                "chunk_type": c.get("chunk_type", "body"),
                "label": c.get("label", "") or "",
            }
            for c in paper_chunks
        ]
        return all_chunks, all_ids

    # Legacy dictionary handling
    all_chunks = []
    all_ids = []
    for title, data in paper_chunks.items():
        chunks = data["chunks"]
        ids = data["ids"]
        for i in range(len(chunks)):
            all_chunks.append(chunks[i])
            all_ids.append(ids[i])

    last_metadatas = [
        {"paper_title": cid.split(".pdf")[0], "section": "N/A", "section_path": "N/A", "chunk_type": "body", "label": ""}
        for cid in all_ids
    ]
    return all_chunks, all_ids


# ─────────────────────────────────────────────────────────────────────────────
# Embedding & Vector Database (GPU Accelerated)
# ─────────────────────────────────────────────────────────────────────────────
_cached_embedding_model: Optional[SentenceTransformer] = None


def load_embedded_model(model_name: str = "BAAI/bge-base-en-v1.5", device: Optional[str] = None) -> SentenceTransformer:
    global _cached_embedding_model
    if _cached_embedding_model is not None:
        return _cached_embedding_model

    if device is None:
        device = get_device()
    logger.info(f"Loading embedding model '{model_name}' on device: {device}")
    try:
        # Prefer cached local weights to avoid unneeded network calls
        _cached_embedding_model = SentenceTransformer(model_name, device=device, local_files_only=True)
    except Exception:
        _cached_embedding_model = SentenceTransformer(model_name, device=device)
    return _cached_embedding_model


def create_embedding_text(embedding_model: SentenceTransformer, chunks: list, size: int = 64):
    try:
        embedding_text = embedding_model.encode(
            chunks,
            batch_size=size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    except RuntimeError as e:
        if "CUDA" in str(e) or "kernel" in str(e) or "device" in str(e):
            logger.warning(f"CUDA kernel execution failed ({e}). Moving embedding model to CPU.")
            embedding_model.to("cpu")
            embedding_text = embedding_model.encode(
                chunks,
                batch_size=size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        else:
            raise e
    return embedding_text, embedding_text.shape


def initialize_vector_database(path: str = CHROMA_PATH) -> chromadb.PersistentClient:
    os.makedirs(path, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def initialize_collections(client: chromadb.PersistentClient, collection_name: str = COLLECTION_NAME):
    return client.create_collection(collection_name)


def get_collection(client: chromadb.PersistentClient, collection_name: str = COLLECTION_NAME):
    return client.get_collection(collection_name)


def add_in_collection(collection, ids: list, chunks: list, embedded_text, metadatas: Optional[list] = None):
    embeddings_list = embedded_text.tolist() if hasattr(embedded_text, "tolist") else embedded_text
    kwargs = {
        "ids": ids,
        "documents": chunks,
        "embeddings": embeddings_list,
    }
    if metadatas is not None:
        kwargs["metadatas"] = metadatas
    collection.add(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval & Prompt Construction
# ─────────────────────────────────────────────────────────────────────────────
def query_builder(user_query: str, top_k: int = 15) -> Tuple[str, list, list]:
    global embedding_model, collection

    if embedding_model is None:
        embedding_model = load_embedded_model()

    if collection is None:
        client = initialize_vector_database()
        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            pass

    if collection is None:
        raise RuntimeError("No indexed documents found in vector store. Please index PDFs first.")

    total_in_db = collection.count()
    actual_top_k = min(top_k, total_in_db) if total_in_db > 0 else top_k

    print(f"[Retrieval] Vector DB contains {total_in_db} total indexed chunks. Searching for top {actual_top_k} most relevant...")

    query_embedding = embedding_model.encode(
        user_query,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    result = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=actual_top_k,
        include=["documents", "metadatas"]
    )

    ids = result["ids"][0]
    chunks = result["documents"][0]
    metas = result["metadatas"][0] if result.get("metadatas") and result["metadatas"] else []

    sources = []
    context = ""

    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        paper_title = meta.get("paper_title")
        if not paper_title:
            paper_title = ids[i].split(".pdf")[0].replace("_", " ")

        section_path = meta.get("section_path") or meta.get("section", "Section")
        chunk_type = meta.get("chunk_type", "body")
        label = meta.get("label", "")

        if paper_title not in sources:
            sources.append(paper_title)

        type_header = f" [{label}]" if label else (f" [{chunk_type}]" if chunk_type in ("table", "code", "figure_caption") else "")

        context += f"""
Paper   : {paper_title}
Section : {section_path}{type_header}
Content : {chunks[i]}

-------"""

    prompt = f"""
You are an expert research assistance helping researchers and students. 
You job is to give precise and clean answer user question clearly and accurately using only the context provided.
if the answer is not in the context, say sorry can't answer it based on given context.
You may ask clarifying or counter-questions to deepen understanding.


User question : 
{user_query}


Context:
{context}
"""
    return prompt, sources, ids


# ─────────────────────────────────────────────────────────────────────────────
# Answer Generation (OpenRouter API)
# ─────────────────────────────────────────────────────────────────────────────
def answer_generator(prompt: str) -> dict:
    load_dotenv()
    api_key = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    llm_model = os.getenv("RAG_LLM_MODEL", "nvidia/nemotron-3.5-super-vl:free")

    if not api_key:
        return {
            "success": False,
            "content": None,
            "reasoning": None,
            "error": "No OpenRouter API key set. Set OPENROUTER_API_KEY in a .env file "
                     "or in Streamlit secrets, then ask again.",
        }

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            extra_body={"reasoning": {"enabled": True}}
        )
        message = response.choices[0].message

        return {
            "success": True,
            "content": message.content,
            "reasoning": getattr(message, "reasoning", "") or "",
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "content": None,
            "reasoning": None,
            "error": str(e)
        }
