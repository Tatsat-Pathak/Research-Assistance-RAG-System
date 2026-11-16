"""
evaluate_literature_review.py — Thematic Benchmark Suite for Literature Review
=============================================================================================
Evaluates the RAG system on the thesis reference papers ("Literature Review"
folders + documents/), comparing TWO versions of the same pipeline on the
SAME index, so the comparison is fair:

  * "original"  — RAG_Pipeline.py / app.py (from-scratch, no LangChain)
  * "langchain" — langchain_rag.py (LangChain wiring, same parsing/chunking)

Workflow (v3 — incremental):
  1. Scan all papers and group by theme (folder name).
  2. Ingest & extract via MinerU with SHA-256 disk caching (parse_cache/).
     Only papers NOT in the cache are parsed with MinerU; the rest load
     from cache in seconds.
  3. Open the existing Chroma collection (do NOT delete it) and work out
     which parsed papers are already indexed, by title. Only the NEW
     papers are chunked, embedded and added. The 30 papers already in
     the collection are never re-parsed, re-embedded or re-indexed.
  4. Retrieval benchmark for both versions — global Top-1/3/5, MRR and
     per-paper coverage (see benchmark_retrieval.py, which owns the
     scoring and the 12 queries). The old version of this file scored
     the per-paper grouped order, which mostly measured where a paper
     sits in the index order, not relevance; that measurement is gone.
  5. End-to-end check for both versions (unless --no-llm): the same
     questions go through each version's full answer path (OpenRouter
     LLM, same model), and the answers are compared on keyword coverage,
     sources returned, and latency.

Three modes:
  * default (incremental): parse only uncached papers, index only
    papers not already in the collection, then benchmark. This is the
    mode for adding new papers — e.g. after downloading ScaNN and
    IVF-RaBitQ from Drive into Literature Review/Core Papers/, only
    those 2 get MinerU-parsed and only their chunks get embedded.
  * --reuse-index: benchmark only. Open the existing collection and
    measure it; do not parse or index anything.
  * --rebuild-index: delete the collection and index everything from
    scratch (clean slate). Use it if the index is corrupted or you
    want to remove papers you deleted from the folder — the incremental
    mode only ADDS papers, it never removes them.

Notes:
  * The 12 queries and the strict scoring live in benchmark_retrieval.py
    (same folder). This file imports it — keep the two files together.
    A chunk counts as a HIT only if its text contains at least 2 of the
    query's expected_keywords AND (its theme is the query's theme OR its
    section is a specific expected section). Generic section names —
    INTRODUCTION, EXPERIMENT, METHOD, ALGORITHM, SEARCH, RESULTS,
    CONCLUSION — do NOT count on their own.
  * Put new papers in the theme subfolder their question expects:
    ScaNN and IVF-RaBitQ belong in "Core Papers" (Q8/Q9 use that
    theme). A paper in the wrong folder gets the wrong theme metadata
    and its question will not hit.
  * First run downloads the bge model (~1 GB) unless it is already
    cached. The file forces HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1
    (restored 2026-11-16 — the original file had them, and dropping
    them broke MinerU parsing on this machine). Only remove them on
    a machine where the models are NOT cached yet.
  * Parsing uses max_workers=1 by default (sequential). RAG_Pipeline
    caps workers at 2 anyway, because each worker loads its own MinerU
    models and can use ~4x the RAM. Use --workers 2 at most.
  * LLM answers are non-deterministic: run the end-to-end check twice
    and the wording will differ. Treat the LLM numbers as a spot-check
    (did it cover the papers? did it use the right keywords?), and the
    retrieval numbers (Top-1/3/5, MRR, coverage) as the stable comparison.
  * The OpenRouter key comes from a local .env file, same as app.py.
    Without it, the LLM step is skipped and retrieval still runs.

Run (on your laptop, in the repo folder, venv active):
  python evaluate_literature_review.py --no-llm            # incremental: add new papers, retrieval only
  python evaluate_literature_review.py                    # incremental + LLM end-to-end step
  python evaluate_literature_review.py --reuse-index --no-llm   # measure the existing index only
  python evaluate_literature_review.py --rebuild-index --no-llm # wipe and re-index everything
  python evaluate_literature_review.py --top-k 15 --versions original
"""

import argparse
import json
import math
import os

# 2026-11-16: restored from the original file — do NOT remove these.
# The old benchmark set these two lines and parsed all 31 papers fine.
# This modified file dropped them, and then 29/31 papers failed in
# MinerU post-processing ("private_ranges" type error) after the models
# themselves ran fine. With offline mode on, huggingface_hub never
# touches the network, so MinerU uses exactly the model files already
# cached on this machine instead of checking the hub for a newer
# revision whose config may not match the installed MinerU code.
# Only remove these on a machine where the bge / MinerU models are NOT
# cached yet — a first run needs the network to download them.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

import RAG_Pipeline as rag
from chunks_generated import chunk_papers

# Scoring, the 12 queries and the retrieval measurement live here —
# single source of truth, shared with benchmark_retrieval.py.
import benchmark_retrieval as br

try:
    import langchain_rag
except ImportError as _lc_err:  # LangChain packages not installed yet
    langchain_rag = None
    _LANGCHAIN_IMPORT_ERROR = _lc_err
else:
    _LANGCHAIN_IMPORT_ERROR = None

# Dedicated collection name for literature review (separate from the
# main app's collection, so benchmarking never clobbers it).
LIT_REVIEW_COLLECTION = "literature_review_chroma_db"
DEFAULT_LLM_MODEL = "nvidia/nemotron-3.5-super-vl:free"


def _get_langchain_vectorstore(model, persist_directory: str, collection_name: str):
    if langchain_rag is None:
        raise RuntimeError(
            f"LangChain version requested but langchain_rag could not be imported "
            f"({_LANGCHAIN_IMPORT_ERROR}). Install the LangChain packages first: "
            f"pip install langchain langchain-chroma langchain-huggingface langchain-openai"
        )
    return langchain_rag.get_vectorstore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embeddings=br._shared_langchain_embeddings(model),
    )


def discover_literature_papers(lit_dir: str = "Literature Review", doc_dir: str = "documents") -> Dict[str, List[Path]]:
    """Discovers all PDFs in Literature Review subfolders and documents/ grouped by theme."""
    categorized: Dict[str, List[Path]] = {}

    lit_path = Path(lit_dir)
    if lit_path.exists():
        for theme_dir in sorted(lit_path.iterdir()):
            if theme_dir.is_dir():
                pdfs = sorted(list(theme_dir.glob("*.pdf")))
                if pdfs:
                    categorized[theme_dir.name] = pdfs

    if doc_dir:
        doc_path = Path(doc_dir)
        if doc_path.exists():
            doc_pdfs = sorted(list(doc_path.glob("*.pdf")))
            if doc_pdfs:
                categorized["General & Additional Research"] = doc_pdfs

    return categorized


def batch_ingest_and_cache(categorized_papers: Dict[str, List[Path]], max_workers: int = 1) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Parses all papers with caching to parse_cache/.
    Returns:
      parsed_docs: {clean_title: {heading: [items...]}}
      paper_to_theme: {clean_title: theme_name}
    max_workers is capped at 2 by RAG_Pipeline (each worker loads its
    own MinerU models); default 1 is the safe, sequential choice.
    """
    print("\n" + "=" * 70)
    print(" 1. THEMATIC BATCH INGESTION & CACHING")
    print("=" * 70)

    total_papers = sum(len(flist) for flist in categorized_papers.values())
    print(f"[*] Total categorized papers found: {total_papers}")
    for theme, flist in categorized_papers.items():
        print(f"    - [{theme}]: {len(flist)} papers")

    all_file_paths = []
    file_to_theme = {}
    for theme, flist in categorized_papers.items():
        for fpath in flist:
            p_str = str(fpath.resolve())
            all_file_paths.append(p_str)
            file_to_theme[p_str] = theme

    # Check cache coverage
    cached_count = 0
    uncached_paths = []
    for p in all_file_paths:
        h = rag.compute_file_hash(p)
        cache_f = rag.PARSE_CACHE_DIR / f"{h}_mineru.json"
        if cache_f.exists():
            cached_count += 1
        else:
            uncached_paths.append(p)

    if total_papers:
        print(f"[*] Cache Status: {cached_count}/{total_papers} cached ({cached_count/total_papers*100:.1f}%)")
    if uncached_paths:
        print(f"[*] Uncached PDFs to process with MinerU ({len(uncached_paths)} files)...")

    t0 = time.perf_counter()
    # Process uncached in batch (result is re-loaded from cache below,
    # so the return value is intentionally not used here).
    if uncached_paths:
        print(f"[*] Parsing with max_workers={max_workers}...")
        rag.parse_documents_batch(uncached_paths, max_workers=max_workers)

    # Now load all papers from cache
    parsed_docs = {}
    paper_to_theme = {}
    for p in all_file_paths:
        try:
            fname, title, sections = rag.parse_document(p)
            if sections:
                doc_key = title if title and title != "untitled" else fname
                parsed_docs[doc_key] = sections
                paper_to_theme[doc_key] = file_to_theme[p]
        except Exception as e:
            print(f"[-] Warning: Failed to load {os.path.basename(p)}: {e}")

    total_time = time.perf_counter() - t0
    print(f"[+] All {len(parsed_docs)} papers ready in memory ({total_time:.2f}s total)")
    return parsed_docs, paper_to_theme


def chunk_and_enrich_literature(parsed_docs: Dict[str, Any], paper_to_theme: Dict[str, str]) -> List[Dict[str, Any]]:
    """Chunks all documents with the production chunker.

    Theme is stored in the chunk dict (and later in Chroma metadata)
    ONLY — it is NOT injected into embed_text. The production pipeline
    does not put the theme in the embedded text, so the benchmark must
    not either, or it would be measuring a different system.
    """
    print("\n" + "=" * 70)
    print(" 2. SEMANTIC CHUNKING & THEME METADATA ENRICHMENT")
    print("=" * 70)

    t0 = time.perf_counter()
    all_chunks = []
    theme_chunk_counts: Dict[str, int] = {}

    for doc_title, sections in parsed_docs.items():
        theme = paper_to_theme.get(doc_title, "General Research")
        doc_input = {doc_title: sections}
        raw_chunks = chunk_papers(doc_input)

        for c in raw_chunks:
            c["theme"] = theme
            all_chunks.append(c)
            theme_chunk_counts[theme] = theme_chunk_counts.get(theme, 0) + 1

    chunk_time = time.perf_counter() - t0
    word_counts = [c.get("word_count", 0) for c in all_chunks]
    mean_words = statistics.mean(word_counts) if word_counts else 0
    median_words = statistics.median(word_counts) if word_counts else 0

    print(f"[+] Total Chunks Generated: {len(all_chunks)} ({chunk_time*1000:.1f} ms)")
    print(f"[+] Chunk Length: Mean = {mean_words:.1f} words, Median = {median_words} words")
    print(f"[*] Chunk Breakdown by Theme:")
    for theme, count in theme_chunk_counts.items():
        pct = (count / len(all_chunks) * 100) if all_chunks else 0
        print(f"    - {theme:<28}: {count:>4} chunks ({pct:5.1f}%)")

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Incremental indexing — only NEW papers are embedded and added.
# The collection is never deleted by this path (see --rebuild-index).
# ─────────────────────────────────────────────────────────────────────────────
def existing_titles_and_ids(collection) -> Tuple[set, set]:
    """Distinct paper titles + chunk ids already in the collection."""
    data = collection.get(include=["metadatas"])
    titles: set = set()
    ids: set = set()
    for cid, meta in zip(data.get("ids") or [], data.get("metadatas") or []):
        ids.add(cid)
        title = (meta or {}).get("paper_title")
        if title:
            titles.add(title)
    return titles, ids


def select_new_docs(parsed_docs: Dict[str, Any], existing_titles: set) -> Dict[str, Any]:
    """Parsed papers whose title is not already in the collection."""
    return {title: sections for title, sections in parsed_docs.items()
            if title not in existing_titles}


def filter_new_chunks(all_chunks: List[Dict[str, Any]], existing_ids: set) -> Tuple[List[Dict[str, Any]], int]:
    """Chunks whose chunk_id is not already in the collection.
    Belt-and-braces under the title check: if a paper was re-parsed
    with a slightly different title, its chunks still will not be
    added twice, because chunk_ids are deterministic per paper."""
    new_chunks = [c for c in all_chunks if c.get("chunk_id") not in existing_ids]
    return new_chunks, len(all_chunks) - len(new_chunks)


def _chunk_metadatas(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "paper_title": c["paper_title"],
            "paper_id": c.get("paper_id", "") or "",
            "theme": c.get("theme", "General"),
            "section": c["section"],
            "section_path": " > ".join(c["section_path"]) if isinstance(c["section_path"], list) else str(c["section_path"]),
            "chunk_type": c.get("chunk_type", "body"),
            "label": c.get("label", "") or "",
        }
        for c in chunks
    ]


def open_or_create_collection(client, collection_name: str = LIT_REVIEW_COLLECTION):
    """Open the existing collection, or create it on the first run.
    Returns (collection, was_created). Never deletes anything."""
    try:
        collection = client.get_collection(collection_name)
        print(f"[+] Opened existing collection '{collection_name}' ({collection.count()} chunks).")
        return collection, False
    except Exception:
        collection = client.create_collection(collection_name)
        print(f"[+] Created new collection '{collection_name}'.")
        return collection, True


def index_new_into_chromadb(collection, new_chunks: List[Dict[str, Any]], model,
                            existing_ids: set, embed_fn=None) -> Tuple[int, int]:
    """Embed + add ONLY new_chunks to an existing collection.
    Returns (n_added, n_skipped_duplicate_ids).
    embed_fn defaults to rag.create_embedding_text(model, texts, size=64);
    pass a fake in tests to avoid loading the model."""
    new_chunks, n_dup = filter_new_chunks(new_chunks, existing_ids)
    if not new_chunks:
        print("[*] No new chunks to index (every chunk id is already present).")
        return 0, n_dup

    texts = [c["embed_text"] for c in new_chunks]
    ids = [c["chunk_id"] for c in new_chunks]
    metadatas = _chunk_metadatas(new_chunks)

    print(f"[*] Embedding {len(texts)} new chunks (papers already indexed are skipped)...")
    t0 = time.perf_counter()
    if embed_fn is None:
        embeddings, shape = rag.create_embedding_text(model, texts, size=64)
    else:
        embeddings, shape = embed_fn(model, texts)
    embed_time = time.perf_counter() - t0
    if embed_time > 0:
        print(f"    [+] Embedding Time: {embed_time:.2f}s ({len(texts)/embed_time:.1f} chunks/sec)")

    # ChromaDB insert in batches of 500
    t0 = time.perf_counter()
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        end_idx = min(i + batch_size, len(ids))
        rag.add_in_collection(
            collection,
            ids=ids[i:end_idx],
            chunks=texts[i:end_idx],
            embedded_text=embeddings[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
    index_time = time.perf_counter() - t0
    if index_time > 0:
        print(f"    [+] Index Insert: {index_time:.3f}s ({len(ids)/index_time:.1f} chunks/sec)")
    print(f"    [+] Collection now holds {collection.count()} chunks.")
    return len(new_chunks), n_dup


def benchmark_end_to_end(collection, model, queries: List[Dict[str, Any]], top_k: int = 10,
                         versions: List[str] = ("original", "langchain"),
                         llm_model: str = DEFAULT_LLM_MODEL) -> Dict[str, Any]:
    """End-to-end check: same questions through each version's full
    answer path. Compared on answer keyword coverage, sources returned,
    and latency — NOT on wording, which changes every run."""
    print("\n" + "=" * 70)
    print(" 5. END-TO-END MULTI-PAPER SYNTHESIS (LLM)")
    print("=" * 70)
    print(" NOTE: LLM wording changes every run. Keyword coverage and")
    print(" source coverage are the comparison; the text itself is a spot-check.")

    rag.embedding_model = model
    rag.collection = collection

    vectorstore = None
    llm = None
    if "langchain" in versions:
        vectorstore = _get_langchain_vectorstore(model, rag.CHROMA_PATH, LIT_REVIEW_COLLECTION)
        llm = langchain_rag.get_llm(model=llm_model)

    results: Dict[str, Any] = {}
    for version in versions:
        print(f"\n --- Version: {version} ---")
        per_query = []
        for item in queries:
            t0 = time.perf_counter()
            answer, sources, error = "", [], None
            try:
                if version == "original":
                    prompt, sources, _ids = rag.query_builder(item["query"], top_k=top_k)
                    ans = rag.answer_generator(prompt)
                    if ans.get("success"):
                        answer = ans.get("content") or ""
                    else:
                        error = ans.get("error", "LLM failed")
                else:
                    res = langchain_rag.ask(item["query"], top_k=top_k, llm=llm, vectorstore=vectorstore)
                    answer = res.get("answer") or ""
                    sources = res.get("sources") or []
            except Exception as e:
                error = str(e)
            lat = time.perf_counter() - t0
            kw_found = [k for k in (item.get("expected_keywords") or []) if br._norm(k) in br._norm(answer)]
            coverage = round(len(kw_found) / len(item["expected_keywords"]) * 100, 1) if item.get("expected_keywords") else 0.0
            print(f" {item['id']}: {len(answer)} chars, {len(sources)} source paper(s), "
                  f"keyword coverage {coverage}%, {lat:.1f}s" + (f"  ERROR: {error}" if error else ""))
            per_query.append({
                "id": item["id"],
                "query": item["query"],
                "answer": answer,
                "sources": list(sources),
                "keyword_coverage_pct": coverage,
                "keywords_found": kw_found,
                "latency_sec": round(lat, 2),
                "error": error,
            })
        results[version] = {"queries": per_query}
    return results


def run_full_literature_benchmark(args) -> None:
    start_all = time.perf_counter()
    print("=" * 70)
    print(" LITERATURE REVIEW RAG SYSTEM — COMPREHENSIVE BENCHMARK")
    print(" (original vs LangChain, one shared index)")
    print("=" * 70)

    versions = ["original", "langchain"] if args.versions == "both" else [args.versions]
    if "langchain" in versions and langchain_rag is None:
        print(f"[-] LangChain requested but langchain_rag failed to import: {_LANGCHAIN_IMPORT_ERROR}")
        print("    Install it first: pip install langchain langchain-chroma langchain-huggingface langchain-openai")
        return

    new_papers_indexed = 0
    new_chunks_indexed = 0
    collection = None
    model = None

    if args.reuse_index:
        print("[*] --reuse-index: opening the existing collection, skipping parse/chunk/embed.")
        client = rag.initialize_vector_database()
        try:
            collection = client.get_collection(LIT_REVIEW_COLLECTION)
        except Exception:
            print(f"[-] No existing collection '{LIT_REVIEW_COLLECTION}' to reuse. Run without --reuse-index first.")
            return
        if collection.count() == 0:
            print(f"[-] Existing collection '{LIT_REVIEW_COLLECTION}' is empty. Run without --reuse-index first.")
            return
        print(f"[+] Reusing {collection.count()} indexed chunks.")
    else:
        # 1. Discover papers
        categorized = discover_literature_papers(args.lit_dir, args.doc_dir)
        if not categorized:
            print(f"[-] No PDFs found in '{args.lit_dir}' or '{args.doc_dir}'.")
            print("    Run this from the repo folder that contains those folders,")
            print("    or pass --lit-dir / --doc-dir with the right paths.")
            return

        # 2. Ingest & cache (only uncached papers are MinerU-parsed)
        workers = min(args.workers, 2)
        if args.workers > 2:
            print(f"[!] --workers {args.workers} capped at 2 (MinerU RAM per worker).")
        parsed_docs, paper_to_theme = batch_ingest_and_cache(categorized, max_workers=workers)
        if not parsed_docs:
            print("[-] No papers could be parsed.")
            return

        # 3. Open the collection (never delete it here) and index only
        #    the papers that are not already in it.
        client = rag.initialize_vector_database()
        if args.rebuild_index:
            try:
                client.delete_collection(LIT_REVIEW_COLLECTION)
                print("[*] --rebuild-index: deleted the old collection; indexing everything from scratch.")
            except Exception:
                pass
        collection, _created = open_or_create_collection(client, LIT_REVIEW_COLLECTION)
        existing_titles, existing_ids = existing_titles_and_ids(collection)
        new_docs = select_new_docs(parsed_docs, existing_titles)
        new_papers_indexed = len(new_docs)
        print(f"[*] {len(parsed_docs)} papers parsed/loaded; {len(existing_titles)} already in the "
              f"collection; {new_papers_indexed} new to index.")
        if new_docs:
            for title in new_docs:
                print(f"    + NEW: {title} [{paper_to_theme.get(title, '')}]")
            new_chunks = chunk_and_enrich_literature(new_docs, paper_to_theme)
            model = rag.load_embedded_model()
            n_added, n_dup = index_new_into_chromadb(collection, new_chunks, model, existing_ids)
            new_chunks_indexed = n_added
            if n_dup:
                print(f"[!] {n_dup} chunks were already present by id and were skipped.")
        else:
            print("[*] Nothing new to index — the collection already has every paper.")

    # The embedding model is needed for the benchmark even when nothing
    # was indexed in this run.
    if model is None:
        model = rag.load_embedded_model()

    papers = br.list_papers(collection)
    total_papers = len(papers)
    total_chunks = collection.count()
    print(f"[+] Collection now holds {total_papers} papers, {total_chunks} chunks.")

    # 4. IR Benchmark (both versions) — global ranking + per-paper
    #    coverage, via benchmark_retrieval.py.
    vectorstore = None
    if "langchain" in versions:
        vectorstore = _get_langchain_vectorstore(model, rag.CHROMA_PATH, LIT_REVIEW_COLLECTION)
    ir_results = br.benchmark_retrieval(collection, model, vectorstore, rag, langchain_rag,
                                        br.BENCHMARK_QUERIES, top_k=args.top_k, versions=versions)

    # 5. End-to-end LLM check (both versions), unless skipped
    llm_results = {}
    if args.no_llm:
        print("\n[*] --no-llm: skipping the end-to-end LLM step.")
    else:
        from dotenv import load_dotenv
        load_dotenv()
        if not (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")):
            print("\n[-] No OPENROUTER_API_KEY in .env — skipping the end-to-end LLM step.")
            print("    Retrieval results below are still valid.")
        else:
            llm_results = benchmark_end_to_end(collection, model, br.BENCHMARK_QUERIES,
                                               top_k=args.top_k, versions=versions,
                                               llm_model=args.llm_model)

    total_elapsed = time.perf_counter() - start_all
    print("\n" + "=" * 70)
    print(" COMPARISON (retrieval — the stable numbers)")
    print("=" * 70)
    print(f"{'Version':<12} {'Top-1':>6} {'Top-3':>6} {'Top-5':>6} {'MRR':>7} {'Covered':>8} {'Glob lat':>9} {'PP lat':>8}")
    for version in versions:
        g, c = ir_results[version]["global"], ir_results[version]["coverage"]
        print(f"{version:<12} {g['top1_acc_pct']:>5.1f}% {g['top3_acc_pct']:>5.1f}% {g['top5_acc_pct']:>5.1f}% "
              f"{g['mrr']:>7.4f} {c['coverage_pct']:>7.1f}% {g['avg_retrieval_lat_ms']:>9.2f} "
              f"{c['avg_retrieval_lat_ms']:>8.2f}")
    print("=" * 70)
    print(f" BENCHMARK COMPLETE (Total Runtime: {total_elapsed:.2f}s)")
    print("=" * 70)

    output_data = {
        "versions": versions,
        "top_k": args.top_k,
        "collection": LIT_REVIEW_COLLECTION,
        "total_papers": total_papers,
        "total_chunks": total_chunks,
        "new_papers_indexed": new_papers_indexed,
        "new_chunks_indexed": new_chunks_indexed,
        "total_runtime_sec": round(total_elapsed, 2),
        "papers": papers,
        "ir_metrics": ir_results,
        "llm_metrics": llm_results,
    }
    out_file = Path(args.output)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"[+] Benchmark results saved to: {out_file.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the RAG pipeline (original vs LangChain) on the thesis reference papers. Incremental by default: only new papers are parsed and indexed.")
    parser.add_argument("--lit-dir", default="Literature Review", help="Folder with theme subfolders of PDFs")
    parser.add_argument("--doc-dir", default="documents", help="Extra folder of PDFs (empty string to skip)")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval budget per query (same for both versions)")
    parser.add_argument("--workers", type=int, default=1, help="MinerU parse workers (max 2)")
    parser.add_argument("--versions", choices=["both", "original", "langchain"], default="both",
                        help="Which pipeline version(s) to test")
    parser.add_argument("--no-llm", action="store_true", help="Retrieval only, no OpenRouter LLM calls")
    parser.add_argument("--reuse-index", action="store_true", help="Benchmark only: open the existing collection, do not parse or index anything")
    parser.add_argument("--rebuild-index", action="store_true", help="Delete the collection and index everything from scratch (clean slate)")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="OpenRouter model for the end-to-end step")
    parser.add_argument("--output", default="literature_review_benchmark_results.json", help="Results JSON file")
    args = parser.parse_args()
    run_full_literature_benchmark(args)


if __name__ == "__main__":
    main()
