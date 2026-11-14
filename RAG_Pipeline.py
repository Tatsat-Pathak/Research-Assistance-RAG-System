"""
RAG_Pipeline.py — Research RAG Pipeline (Streamlit Cloud compatible)
=====================================================================
Uses pymupdf4llm for PDF parsing (lightweight, no GPU required).
Includes diversity-aware retrieval to ensure multi-paper coverage.
"""

import os
import re
import json
import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./upload_pdfs_chroma_db"
COLLECTION_NAME = "chroma_db_documents"
PARSE_CACHE_DIR = Path(os.getenv("RAG_PARSE_CACHE_DIR", "./parse_cache"))
PARSE_CACHE_DIR.mkdir(exist_ok=True)

# Disable ChromaDB anonymous telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Injected at runtime by app.py
embedding_model = None
collection = None
last_metadatas = None


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for cache invalidation."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Document Parsing (pymupdf4llm — lightweight, no GPU)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_with_pymupdf4llm(file_path: str) -> Tuple[str, Dict[str, list]]:
    """
    Parse a PDF using pymupdf4llm → markdown, then split by headings.
    Returns (document_title, sections_dict).
    sections_dict: {heading: [{"type": "text", "content": str}, ...]}
    """
    import pymupdf4llm

    md_text = pymupdf4llm.to_markdown(file_path)
    if not md_text or not md_text.strip():
        return os.path.splitext(os.path.basename(file_path))[0], {}

    lines = md_text.split("\n")

    # Extract title from first heading or filename
    title = os.path.splitext(os.path.basename(file_path))[0]
    for line in lines[:20]:
        line_s = line.strip()
        if line_s.startswith("# ") and len(line_s) > 3:
            title = line_s.lstrip("# ").strip()
            break

    # Split content by markdown headings
    heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$")
    sections: Dict[str, list] = {}
    current_heading = "Introduction"
    current_text_parts: List[str] = []

    # Skip content before first heading that looks like metadata
    started = False

    for line in lines:
        m = heading_pattern.match(line.strip())
        if m:
            # Save previous section
            if current_text_parts and started:
                text = "\n".join(current_text_parts).strip()
                if text:
                    # Deduplicate heading key
                    key = m.group(2).strip()
                    # But first, save current section
                    if current_heading not in sections:
                        sections[current_heading] = []
                    sections[current_heading].append({
                        "type": "text",
                        "content": text
                    })
                    current_text_parts = []

            elif current_text_parts and not started:
                # This is pre-heading content (could be abstract)
                text = "\n".join(current_text_parts).strip()
                if len(text.split()) >= 50:
                    sections["Abstract"] = [{"type": "text", "content": text}]
                current_text_parts = []

            started = True
            current_heading = m.group(2).strip()
            # Clean heading
            current_heading = re.sub(r"^\*+|\*+$", "", current_heading).strip()
            if not current_heading:
                current_heading = "Section"
        else:
            stripped = line.strip()
            # Skip empty lines and horizontal rules
            if stripped and stripped != "---" and stripped != "***":
                current_text_parts.append(stripped)

    # Save last section
    if current_text_parts and started:
        text = "\n".join(current_text_parts).strip()
        if text:
            if current_heading not in sections:
                sections[current_heading] = []
            sections[current_heading].append({
                "type": "text",
                "content": text
            })

    # If no headings were found, create a single section from entire content
    if not sections:
        full_text = "\n".join(l.strip() for l in lines if l.strip()).strip()
        if full_text:
            sections["Full Document"] = [{"type": "text", "content": full_text}]

    return title, sections


def parse_document(file_path: str) -> Tuple[str, str, Dict[str, list]]:
    """
    Parse a PDF file with disk-caching support.
    Returns: (file_name, document_title, sections_dict)
    """
    file_name = os.path.basename(file_path)
    fallback_title = os.path.splitext(file_name)[0]

    # Check parse cache
    file_hash = compute_file_hash(file_path)
    cache_file = PARSE_CACHE_DIR / f"{file_hash}_pymupdf.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            title = cached.get("title", fallback_title)
            sections = cached.get("sections", {})
            # Convert sections back to list-of-dicts format
            for key in sections:
                if isinstance(sections[key], list) and sections[key] and isinstance(sections[key][0], str):
                    sections[key] = [{"type": "text", "content": t} for t in sections[key]]
            return file_name, title, sections
        except Exception as e:
            logger.warning(f"Failed reading cache for {file_name}: {e}. Reparsing...")

    # Parse with pymupdf4llm
    try:
        title, sections = _parse_with_pymupdf4llm(file_path)
    except Exception as e:
        logger.error(f"Failed to parse {file_name}: {e}")
        return file_name, fallback_title, {}

    if not title or title == "untitled":
        title = fallback_title

    # Save to disk cache
    try:
        cache_data = {
            "title": title,
            "sections": {k: [item["content"] if isinstance(item, dict) else item for item in v]
                         for k, v in sections.items()}
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed writing cache for {file_name}: {e}")

    return file_name, title, sections


def parse_documents_batch(file_paths: List[str]) -> Dict[str, dict]:
    """
    Parse multiple PDF files sequentially.
    Returns: {document_title: sections_dict}
    """
    if not file_paths:
        return {}

    results: Dict[str, dict] = {}
    for i, fp in enumerate(file_paths, 1):
        try:
            file_name, title, sections = parse_document(fp)
            if sections:
                key = title if title and title != "untitled" else file_name
                results[key] = sections
                section_count = len(sections)
                item_count = sum(len(v) for v in sections.values())
                print(f"[{i}/{len(file_paths)}] Parsed '{file_name}' → title='{key}', {section_count} sections, {item_count} items")
            else:
                print(f"[{i}/{len(file_paths)}] WARNING: '{file_name}' produced no sections")
        except Exception as exc:
            logger.error(f"Failed to parse {fp}: {exc}")
            print(f"[{i}/{len(file_paths)}] ERROR parsing '{os.path.basename(fp)}': {exc}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────
def chunks_generation(papers_section_heading: dict, chunk_size: int = 500) -> list:
    """
    Generate semantic chunks from parsed papers.
    Input: {paper_title: {heading: [{"type": "text", "content": str}, ...]}}
    Output: list of chunk dicts with paper_title, section, embed_text, chunk_id, etc.
    """
    try:
        from chunks_generated import chunk_papers
        return chunk_papers(papers_section_heading)
    except ImportError:
        logger.info("chunks_generated module not available, using built-in chunker")
        return _builtin_chunk_papers(papers_section_heading, chunk_size)


def _builtin_chunk_papers(papers_section_heading: dict, chunk_size: int = 500) -> list:
    """
    Built-in chunker when chunks_generated.py is not available.
    Splits text into overlapping chunks by word count.
    """
    all_chunks = []
    overlap = 50  # word overlap between chunks

    for paper_title, sections in papers_section_heading.items():
        chunk_idx = 0
        safe_title = re.sub(r"[^\w\-.]", "_", paper_title).strip("_")

        for heading, items in sections.items():
            # Collect all text from items in this section
            section_text_parts = []
            for item in items:
                if isinstance(item, dict):
                    section_text_parts.append(item.get("content", ""))
                elif isinstance(item, str):
                    section_text_parts.append(item)

            section_text = "\n\n".join(t for t in section_text_parts if t.strip())
            if not section_text.strip():
                continue

            words = section_text.split()
            if not words:
                continue

            # Split into chunks
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end])

                safe_heading = re.sub(r"[^\w\-.]", "_", heading).strip("_")
                chunk_id = f"{safe_title}.pdf_{safe_heading}_{chunk_idx}_chunk_{chunk_idx}"

                embed_text = f"Paper: {paper_title}\nSection: {heading}\n\n{chunk_text}"

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "paper_title": paper_title,
                    "section": heading,
                    "section_path": [heading],
                    "chunk_type": "body",
                    "chunk_index": chunk_idx,
                    "word_count": len(chunk_text.split()),
                    "text": chunk_text,
                    "embed_text": embed_text,
                    "label": "",
                })
                chunk_idx += 1

                # Move start forward, with overlap
                if end >= len(words):
                    break
                start = end - overlap

    return all_chunks


def combining_chunks(paper_chunks) -> Tuple[list, list]:
    """
    Flatten and prepare chunks for embedding and ChromaDB indexing.
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
        {"paper_title": cid.split(".pdf")[0].replace("_", " "), "section": "N/A",
         "section_path": "N/A", "chunk_type": "body", "label": ""}
        for cid in all_ids
    ]
    return all_chunks, all_ids


# ─────────────────────────────────────────────────────────────────────────────
# Embedding & Vector Database
# ─────────────────────────────────────────────────────────────────────────────
_cached_embedding_model: Optional[SentenceTransformer] = None


def load_embedded_model(model_name: str = "BAAI/bge-base-en-v1.5", device: Optional[str] = None) -> SentenceTransformer:
    global _cached_embedding_model
    if _cached_embedding_model is not None:
        return _cached_embedding_model

    if device is None:
        device = "cpu"
    logger.info(f"Loading embedding model '{model_name}' on device: {device}")
    try:
        _cached_embedding_model = SentenceTransformer(model_name, device=device, local_files_only=True)
    except Exception:
        _cached_embedding_model = SentenceTransformer(model_name, device=device)
    return _cached_embedding_model


def create_embedding_text(embedding_model: SentenceTransformer, chunks: list, size: int = 64):
    embedding_text = embedding_model.encode(
        chunks,
        batch_size=size,
        normalize_embeddings=True,
        show_progress_bar=False
    )
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
# Diversity-Aware Retrieval & Prompt Construction
# ─────────────────────────────────────────────────────────────────────────────
def _get_indexed_paper_titles() -> List[str]:
    """Get all unique paper titles from the indexed collection."""
    global collection
    if collection is None:
        return []
    try:
        all_data = collection.get(include=["metadatas"])
        titles = set()
        if all_data and all_data.get("metadatas"):
            for meta in all_data["metadatas"]:
                if meta and meta.get("paper_title"):
                    titles.add(meta["paper_title"])
        return list(titles)
    except Exception:
        return []


def _diversity_rerank(result: dict, top_k: int) -> Tuple[list, list, list]:
    """
    Re-rank retrieval results to ensure all indexed papers are represented.
    Uses round-robin selection across papers.
    
    Returns: (ids, chunks, metadatas) — re-ranked lists.
    """
    ids = result["ids"][0]
    chunks = result["documents"][0]
    metas = result["metadatas"][0] if result.get("metadatas") and result["metadatas"] else []

    if not ids:
        return [], [], []

    # Group chunks by paper title
    paper_groups: Dict[str, list] = defaultdict(list)
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        paper_title = meta.get("paper_title", "")
        if not paper_title:
            paper_title = ids[i].split(".pdf")[0].replace("_", " ")
        paper_groups[paper_title].append(i)

    num_papers = len(paper_groups)
    if num_papers <= 1:
        # Only one paper, no diversity needed
        return ids[:top_k], chunks[:top_k], metas[:top_k] if metas else []

    print(f"[Diversity] Found chunks from {num_papers} papers: {list(paper_groups.keys())}")

    # Round-robin selection: ensure each paper gets minimum representation
    min_per_paper = max(2, top_k // num_papers)
    selected_indices = []
    used = set()

    # Phase 1: Guarantee minimum chunks per paper (in their original relevance order)
    for paper_title, indices in paper_groups.items():
        count = 0
        for idx in indices:
            if count >= min_per_paper:
                break
            if idx not in used:
                selected_indices.append(idx)
                used.add(idx)
                count += 1

    # Phase 2: Fill remaining slots with highest-relevance unused chunks
    for i in range(len(ids)):
        if len(selected_indices) >= top_k:
            break
        if i not in used:
            selected_indices.append(i)
            used.add(i)

    # Trim to top_k
    selected_indices = selected_indices[:top_k]

    # Build re-ranked outputs
    new_ids = [ids[i] for i in selected_indices]
    new_chunks = [chunks[i] for i in selected_indices]
    new_metas = [metas[i] for i in selected_indices] if metas else []

    return new_ids, new_chunks, new_metas


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
    # Fetch more than top_k to enable diversity re-ranking
    fetch_k = min(max(top_k * 3, 30), total_in_db) if total_in_db > 0 else top_k

    print(f"[Retrieval] Vector DB contains {total_in_db} total indexed chunks. Fetching top {fetch_k} for diversity re-ranking → final {top_k}...")

    query_embedding = embedding_model.encode(
        user_query,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    result = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=fetch_k,
        include=["documents", "metadatas"]
    )

    # Apply diversity-aware re-ranking
    ids, chunks, metas = _diversity_rerank(result, top_k)

    sources = []
    context = ""

    # Track per-paper chunk counts for logging
    paper_chunk_counts: Dict[str, int] = defaultdict(int)

    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        paper_title = meta.get("paper_title")
        if not paper_title:
            paper_title = ids[i].split(".pdf")[0].replace("_", " ")

        paper_chunk_counts[paper_title] += 1

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

    # Log per-paper distribution
    for pt, cnt in paper_chunk_counts.items():
        print(f"  → Paper '{pt}': {cnt} chunks in context")

    # Build paper list for the prompt
    paper_list = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(sources))

    prompt = f"""You are an expert research assistant helping researchers and students.
Your job is to give precise, clean, and comprehensive answers using ONLY the context provided below.

IMPORTANT INSTRUCTIONS:
- The context contains chunks from {len(sources)} different research paper(s).
- You MUST synthesize information from ALL papers present in the context, not just one.
- When discussing contributions, methods, or findings, explicitly mention which paper each point comes from.
- If the user asks about "these papers" or "the papers", address ALL of them.
- If the answer is not in the context, say you cannot answer based on the given context.
- You may ask clarifying or counter-questions to deepen understanding.

Papers in context:
{paper_list}

User question:
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
