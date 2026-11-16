"""
benchmark_retrieval.py — Retrieval-ONLY benchmark (original vs LangChain)
=========================================================================
Runs against the collection that is ALREADY built. No parsing, no
chunking, no embedding of documents, no indexing. It opens the existing
Chroma collection, lists the papers in it, and measures retrieval only.

Why this file exists (read before trusting the numbers):
  The full benchmark (evaluate_literature_review.py) scored the output
  of query_builder / retrieve_per_paper. Both return chunks GROUPED BY
  PAPER, in the order the papers were indexed — not ranked by how
  relevant they are. So its "match_rank" mostly measured where the
  target paper sits in the index order (auto-tuning papers are first,
  so Q4 scored rank 1; DiskANN sits late, so Q5 scored rank 28),
  not how well retrieval works. The app does not care about that order
  (the LLM reads every paper's chunks), so those ranks understated it.

  This file measures two things that are actually meaningful:
  1. GLOBAL RANKING — one similarity search across ALL papers, top_k
     chunks ranked by embedding similarity. Top-1/3/5 and MRR on this
     ranking say whether the embedding finds the right chunks first.
     Same top_k, same collection, same embeddings for both versions,
     so the comparison is fair.
  2. PER-PAPER COVERAGE — the per-paper set the app really sends to
     the LLM (query_builder / retrieve_per_paper: top chunks per paper
     + each paper's Abstract). Coverage = fraction of questions where
     that set contains at least one qualifying chunk. This is the
     number that decides whether the app can answer the question.

Scoring (same strict rule as the full benchmark, both versions):
  A chunk counts as a HIT only if its text contains at least 2 of the
  query's expected_keywords AND (its theme is the query's theme OR its
  section is a specific expected section). Generic section names —
  INTRODUCTION, EXPERIMENT, METHOD, ALGORITHM, SEARCH, RESULTS,
  CONCLUSION — do NOT count on their own. If a query sets
  "expected_paper_hint", the hit must come from a paper whose title
  contains that hint.

Queries: 12 (the original 6, unchanged, plus 6 new ones for IVF-PQ,
ScaNN, RaBitQ, OPQ, ANN benchmarking, and the quantization survey —
the papers the first 6 never asked about). 6 questions was too few:
each one was worth 16.7 points, so one miss moved Top-1 by a sixth.

Run (on your laptop, in the repo folder, venv active):
  python benchmark_retrieval.py
  python benchmark_retrieval.py --top-k 15 --versions original
  python benchmark_retrieval.py --output retrieval_benchmark_results.json

It opens ./upload_pdfs_chroma_db / collection literature_review_chroma_db
(the collection the full benchmark built). If you run it from another
folder, pass --chroma-dir with the full path to that folder.
Takes about 1-2 minutes: only the embedding model loads (~30 s),
then 12 questions x 2 versions x (1 global + 1 per-paper) retrieval.
No API calls, no .env needed.
"""

import argparse
import json
import os

# Models are already cached on this machine; keep the network out of it
# (same two lines as the full benchmark — see its header for why).
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

DEFAULT_CHROMA_DIR = "./upload_pdfs_chroma_db"
DEFAULT_COLLECTION = "literature_review_chroma_db"

# ─────────────────────────────────────────────────────────────────────────────
# Queries (12). Q1-Q6 are unchanged from evaluate_literature_review.py.
# Q7-Q12 cover the papers the first six never asked about.
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK_QUERIES = [
    {
        "id": "Q1_HNSW_GRAPH",
        "theme": "Foundations Papers",
        "query": "How does Hierarchical Navigable Small World (HNSW) graph construction enable logarithmic search complexity for approximate nearest neighbors?",
        "expected_keywords": ["hnsw", "hierarchical", "navigable", "small world", "logarithmic", "layer", "graph"],
        "expected_sections": ["HNSW", "Navigable Small World", "CONSTRUCTION", "SEARCH ALGORITHM", "HIERARCHICAL", "INTRODUCTION", "ALGORITHM"]
    },
    {
        "id": "Q2_GPU_CAGRA",
        "theme": "GPU Papers",
        "query": "How does CAGRA achieve highly parallel graph construction and approximate nearest neighbor search on GPUs?",
        "expected_keywords": ["cagra", "gpu", "parallel", "warp", "graph", "throughput", "cuda"],
        "expected_sections": ["CAGRA", "GPU", "PARALLEL", "GRAPH CONSTRUCTION", "SEARCH", "EXPERIMENT", "INTRODUCTION"]
    },
    {
        "id": "Q3_QUANTIZATION_PQ",
        "theme": "Core Papers",
        "query": "What is the difference between Product Quantization (PQ), OPQ, and Variance-Aware Quantization in vector similarity search?",
        "expected_keywords": ["product quantization", "quantization", "variance", "subspace", "centroid", "distortion"],
        "expected_sections": ["QUANTIZATION", "PRODUCT QUANTIZATION", "VARIANCE", "EXPERIMENT", "METHOD", "ALGORITHM"]
    },
    {
        "id": "Q4_AUTO_TUNING",
        "theme": "Auto Tunning Papers",
        "query": "How can hyperparameter tuning for HNSW and nearest neighbor search be automated using constrained optimization?",
        "expected_keywords": ["auto", "tuning", "hyperparameter", "constrained optimization", "configuration", "hnsw"],
        "expected_sections": ["AUTOMATING", "OPTIMIZATION", "CONSTRAINED", "HYPERPARAMETER", "TUNING", "EXPERIMENT"]
    },
    {
        "id": "Q5_DISKANN_EDGE",
        "theme": "Resource Constraint",
        "query": "How does DiskANN achieve billion-scale nearest neighbor search on a single node or resource-constrained edge device?",
        "expected_keywords": ["diskann", "ssd", "single node", "billion", "edge", "memory", "vamana"],
        "expected_sections": ["DISKANN", "VAMANA", "SINGLE NODE", "SSD", "MEMORY", "EXPERIMENT", "RESOURCE"]
    },
    {
        "id": "Q6_SERVER_BENCHMARK",
        "theme": "Server Grade Benchmarking",
        "query": "What metrics and benchmarking frameworks are used to evaluate vector databases in server-grade high-dimensional similarity search?",
        "expected_keywords": ["benchmark", "vector database", "scalability", "qps", "latency", "throughput", "recall"],
        "expected_sections": ["BENCHMARK", "FRAMEWORK", "EVALUATION", "SERVER", "THROUGHPUT", "METRICS"]
    },
    {
        "id": "Q7_IVF_PQ",
        "theme": "Core Papers",
        "query": "How does IVF-PQ combine an inverted file index with product quantization to search large vector collections?",
        "expected_keywords": ["ivf", "inverted file", "product quantization", "pq", "nlist", "nprobe", "coarse quantizer"],
        "expected_sections": ["IVF", "INVERTED", "PRODUCT QUANTIZATION", "PQ", "INDEX"]
    },
    {
        "id": "Q8_SCANN",
        "theme": "Core Papers",
        "query": "How does ScaNN use anisotropic vector quantization and asymmetric hashing for fast maximum inner product search?",
        "expected_keywords": ["scann", "anisotropic", "asymmetric", "maximum inner product", "quantization", "hashing"],
        "expected_sections": ["SCANN", "ANISOTROPIC", "ASYMMETRIC", "MAXIMUM INNER PRODUCT"],
        # Hint fixed 2026-11-16: the paper's real title is
        # "Accelerating Large-Scale Inference with Anisotropic Vector
        # Quantization" — it does NOT contain "ScaNN", so the old hint
        # could never be satisfied and Q8 always missed even though the
        # paper was retrieved. The hint is now the title's distinctive
        # phrase, which only this paper has.
        "expected_paper_hint": "Anisotropic Vector Quantization"
    },
    {
        "id": "Q9_RABITQ",
        "theme": "Core Papers",
        "query": "What is RaBitQ and how does it quantize vectors for fast approximate nearest neighbor search?",
        "expected_keywords": ["rabitq", "quantization", "binary", "randomized", "rotation"],
        "expected_sections": ["RABITQ", "QUANTIZATION", "BINARY"],
        "expected_paper_hint": "RaBitQ"
    },
    {
        "id": "Q10_OPQ",
        "theme": "Core Papers",
        "query": "How does Optimized Product Quantization (OPQ) improve on Product Quantization by learning a rotation of the space?",
        "expected_keywords": ["opq", "optimized product quantization", "rotation", "parametric", "product quantization"],
        "expected_sections": ["OPQ", "OPTIMIZED PRODUCT QUANTIZATION", "ROTATION"],
        "expected_paper_hint": "Optimized Product Quantization"
    },
    {
        "id": "Q11_ANN_BENCH",
        "theme": "Server Grade Benchmarking",
        "query": "How are approximate nearest neighbor algorithms benchmarked, and which metrics (recall, QPS, latency) are used to compare them?",
        "expected_keywords": ["benchmark", "recall", "queries per second", "qps", "latency", "throughput"],
        "expected_sections": ["BENCHMARK", "EVALUATION", "FRAMEWORK", "METRICS"]
    },
    {
        "id": "Q12_QUANT_SURVEY",
        "theme": "Foundations Papers",
        "query": "What are the main vector quantization methods for approximate nearest neighbor search and how do they compare?",
        "expected_keywords": ["quantization", "product quantization", "scalar quantization", "binary quantization", "survey"],
        "expected_sections": ["QUANTIZATION", "SURVEY", "PRODUCT QUANTIZATION"]
    }
]

# Section names so generic that matching on them alone proves nothing.
GENERIC_SECTIONS = {
    "introduction", "experiment", "experiments", "method", "methods",
    "methodology", "algorithm", "algorithms", "search", "results",
    "conclusion", "conclusions", "related work", "background",
}


def _norm(text: str) -> str:
    """Lowercase, punctuation/underscores -> single spaces, for matching."""
    return re.sub(r"[\W_]+", " ", str(text or "")).lower().strip()


def _is_hit(item: Dict[str, Any], query: Dict[str, Any]) -> Tuple[bool, int]:
    """Is this chunk a HIT for this query? Returns (hit, keywords found).

    HIT = text contains >= 2 expected_keywords AND (theme matches the
    query's theme OR section is a specific expected section) AND (if the
    query sets expected_paper_hint, the paper title contains it).
    """
    expected_kw = query.get("expected_keywords") or []
    expected_sec = [_norm(s) for s in (query.get("expected_sections") or [])]
    specific_sec = [s for s in expected_sec if s and s not in GENERIC_SECTIONS]
    hint = _norm(query.get("expected_paper_hint", ""))

    text = _norm(item.get("text", ""))
    kw_found = [k for k in expected_kw if _norm(k) in text]
    if len(kw_found) < 2:
        return False, len(kw_found)
    if hint and hint not in _norm(item.get("paper_title", "")):
        return False, len(kw_found)
    theme_ok = _norm(item.get("theme", "")) == _norm(query.get("theme", ""))
    section_text = _norm(f"{item.get('section_path', '')} {item.get('section', '')}")
    section_ok = any(s in section_text for s in specific_sec)
    if theme_ok or section_ok:
        return True, len(kw_found)
    return False, len(kw_found)


def score_retrieved(items: List[Dict[str, Any]], query: Dict[str, Any]) -> Tuple[Optional[int], int]:
    """Rank of the first HIT in `items` (1-based), and the best keyword
    count found in any chunk. `items` must already be in rank order —
    for the global ranking that means similarity order."""
    best_kw = 0
    for rank, item in enumerate(items, 1):
        hit, kw_found = _is_hit(item, query)
        best_kw = max(best_kw, kw_found)
        if hit:
            return rank, kw_found
    return None, best_kw


def count_hits(items: List[Dict[str, Any]], query: Dict[str, Any]) -> int:
    """How many chunks in `items` qualify as hits (order-independent)."""
    return sum(1 for item in items if _is_hit(item, query)[0])


def _summarise_global(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Top-1/3/5 + MRR from the global match ranks."""
    n = len(per_query)
    if n == 0:
        return {"top1_acc_pct": 0.0, "top3_acc_pct": 0.0, "top5_acc_pct": 0.0,
                "mrr": 0.0, "avg_retrieval_lat_ms": 0.0}
    ranks = [q["global_match_rank"] for q in per_query]
    top1 = sum(1 for r in ranks if r == 1) / n * 100
    top3 = sum(1 for r in ranks if r and r <= 3) / n * 100
    top5 = sum(1 for r in ranks if r and r <= 5) / n * 100
    mrr = sum(1.0 / r for r in ranks if r) / n
    avg_lat = statistics.mean([q["global_latency_ms"] for q in per_query]) if per_query else 0.0
    return {
        "top1_acc_pct": round(top1, 1),
        "top3_acc_pct": round(top3, 1),
        "top5_acc_pct": round(top5, 1),
        "mrr": round(mrr, 4),
        "avg_retrieval_lat_ms": round(avg_lat, 2),
    }


def _summarise_coverage(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Coverage of the per-paper set the app really sends to the LLM."""
    n = len(per_query)
    if n == 0:
        return {"coverage_pct": 0.0, "avg_qualifying_chunks": 0.0, "avg_retrieval_lat_ms": 0.0}
    covered = sum(1 for q in per_query if q["covered"])
    avg_hits = statistics.mean([q["qualifying_chunks"] for q in per_query])
    avg_lat = statistics.mean([q["coverage_latency_ms"] for q in per_query]) if per_query else 0.0
    return {
        "coverage_pct": round(covered / n * 100, 1),
        "avg_qualifying_chunks": round(avg_hits, 2),
        "avg_retrieval_lat_ms": round(avg_lat, 2),
    }


def _to_item(doc: str, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    meta = meta or {}
    return {
        "paper_title": meta.get("paper_title", ""),
        "theme": meta.get("theme", ""),
        "section": meta.get("section", ""),
        "section_path": meta.get("section_path", ""),
        "text": doc or "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval — global ranking (the fix) and per-paper set (what the app uses)
# ─────────────────────────────────────────────────────────────────────────────
def _original_global_items(collection, model, query_text: str, top_k: int) -> List[Dict[str, Any]]:
    """ORIGINAL, global: one similarity search across all papers, ranked
    by embedding similarity. This is the ranking the benchmark scores."""
    query_vec = model.encode(query_text, normalize_embeddings=True,
                             show_progress_bar=False).tolist()
    result = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas"],
    )
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    return [_to_item(d, m) for d, m in zip(docs, metas)]


def _langchain_global_items(vectorstore, query_text: str, top_k: int) -> List[Dict[str, Any]]:
    """LANGCHAIN, global: same one-search ranking via the LangChain wrapper."""
    docs = vectorstore.similarity_search(query_text, k=top_k)
    return [_to_item(d.page_content, d.metadata) for d in docs]


def _original_per_paper_items(rag, collection, query_text: str, top_k: int) -> List[Dict[str, Any]]:
    """ORIGINAL, per-paper: query_builder's grouped set (top chunks per
    paper + each paper's Abstract) — exactly what app.py sends the LLM."""
    _, _, ids = rag.query_builder(query_text, top_k=top_k)
    if not ids:
        return []
    data = collection.get(ids=ids, include=["documents", "metadatas"])
    by_id = {}
    for cid, doc, meta in zip(data.get("ids") or [], data.get("documents") or [],
                              data.get("metadatas") or []):
        by_id[cid] = _to_item(doc, meta)
    return [by_id[cid] for cid in ids if cid in by_id]


def _langchain_per_paper_items(langchain_rag, vectorstore, query_text: str, top_k: int) -> List[Dict[str, Any]]:
    """LANGCHAIN, per-paper: retrieve_per_paper's grouped set."""
    per_paper = langchain_rag.retrieve_per_paper(vectorstore, query_text, top_k=top_k)
    items = []
    for title, docs in per_paper.items():
        for doc in docs:
            items.append(_to_item(doc.page_content, doc.metadata))
    return items


def _shared_langchain_embeddings(model):
    """A LangChain Embeddings object that calls the ALREADY-LOADED bge
    model instead of loading a second copy (fair comparison, ~500 MB saved)."""
    from langchain_core.embeddings import Embeddings

    class SharedBGEEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

        def embed_query(self, text):
            return model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()

    return SharedBGEEmbeddings()


def list_papers(collection) -> List[Dict[str, Any]]:
    """Papers in the collection, in first-seen (index) order, with chunk
    counts. Printed at the start so a missing paper is visible."""
    data = collection.get(include=["metadatas"])
    papers: Dict[str, Dict[str, Any]] = {}
    for meta in (data.get("metadatas") or []):
        meta = meta or {}
        title = meta.get("paper_title") or "Unknown paper"
        entry = papers.setdefault(title, {"title": title, "theme": meta.get("theme", ""), "chunks": 0})
        entry["chunks"] += 1
    return list(papers.values())


def benchmark_retrieval(collection, model, vectorstore, rag, langchain_rag,
                        queries: List[Dict[str, Any]], top_k: int,
                        versions: List[str]) -> Dict[str, Any]:
    """Global ranking + per-paper coverage for each version, same queries,
    same collection, same embeddings, same top_k."""
    print("\n" + "=" * 70)
    print(f" RETRIEVAL BENCHMARK (global Top-{top_k} ranking + per-paper coverage)")
    print("=" * 70)

    # query_builder reads these two singletons in RAG_Pipeline.
    rag.embedding_model = model
    rag.collection = collection

    results: Dict[str, Any] = {}
    for version in versions:
        print(f"\n --- Version: {version} ---")
        print(f"{'#':<3} {'Query':<22} {'Global rank':<11} {'Covered':<8} {'Hits in set':<11} {'Lat (ms)':<9}")
        print("-" * 70)
        per_query = []
        for idx, item in enumerate(queries, 1):
            # 1. Global ranking (the fair comparison)
            t0 = time.perf_counter()
            if version == "original":
                global_items = _original_global_items(collection, model, item["query"], top_k)
            else:
                global_items = _langchain_global_items(vectorstore, item["query"], top_k)
            global_lat = (time.perf_counter() - t0) * 1000
            global_rank, kw_found = score_retrieved(global_items, item)

            # 2. Per-paper set (what the app really sends the LLM)
            t0 = time.perf_counter()
            if version == "original":
                pp_items = _original_per_paper_items(rag, collection, item["query"], top_k)
            else:
                pp_items = _langchain_per_paper_items(langchain_rag, vectorstore, item["query"], top_k)
            coverage_lat = (time.perf_counter() - t0) * 1000
            hits = count_hits(pp_items, item)

            rank_str = str(global_rank) if global_rank else "miss"
            print(f"{idx:<3} {item['id'][:21]:<22} {rank_str:<11} "
                  f"{'yes' if hits else 'no':<8} {hits:<11} {global_lat:.0f}/{coverage_lat:.0f}")
            per_query.append({
                "id": item["id"],
                "query": item["query"],
                "theme": item["theme"],
                "global_match_rank": global_rank,
                "keywords_found": kw_found,
                "covered": bool(hits),
                "qualifying_chunks": hits,
                "per_paper_chunks_returned": len(pp_items),
                "global_latency_ms": round(global_lat, 2),
                "coverage_latency_ms": round(coverage_lat, 2),
                "retrieved_papers": list(dict.fromkeys(r["paper_title"] for r in global_items[:5])),
            })
        results[version] = {
            "global": _summarise_global(per_query),
            "coverage": _summarise_coverage(per_query),
            "queries": per_query,
        }
        g, c = results[version]["global"], results[version]["coverage"]
        print("-" * 70)
        print(f"    [+] Global   Top-1: {g['top1_acc_pct']:.1f}%  Top-3: {g['top3_acc_pct']:.1f}%  "
              f"Top-5: {g['top5_acc_pct']:.1f}%  MRR: {g['mrr']:.4f}  Avg lat: {g['avg_retrieval_lat_ms']:.2f} ms")
        print(f"    [+] Coverage: {c['coverage_pct']:.1f}% of questions have a qualifying chunk in the "
              f"per-paper set (avg {c['avg_qualifying_chunks']:.1f} qualifying chunks, "
              f"avg lat {c['avg_retrieval_lat_ms']:.2f} ms)")
    return results


def run_retrieval_benchmark(args) -> None:
    start_all = time.perf_counter()
    print("=" * 70)
    print(" RETRIEVAL-ONLY BENCHMARK — original vs LangChain")
    print(" (existing collection, no parsing, no indexing)")
    print("=" * 70)

    # Heavy imports live here, not at module level, so the scoring
    # helpers above stay importable (and testable) without them.
    import RAG_Pipeline as rag

    try:
        import langchain_rag
        langchain_import_error = None
    except ImportError as exc:
        langchain_rag = None
        langchain_import_error = exc

    versions = ["original", "langchain"] if args.versions == "both" else [args.versions]
    if "langchain" in versions and langchain_rag is None:
        print(f"[-] LangChain requested but langchain_rag failed to import: {langchain_import_error}")
        print("    Install it first: pip install langchain langchain-chroma langchain-huggingface langchain-openai")
        return

    # Open the EXISTING collection. Never create or index here: if it is
    # missing or empty, that is an error, not a reason to rebuild it.
    client = rag.initialize_vector_database(args.chroma_dir)
    try:
        collection = client.get_collection(args.collection)
    except Exception:
        print(f"[-] No collection '{args.collection}' in '{args.chroma_dir}'.")
        print("    Run the full benchmark once (evaluate_literature_review.py)")
        print("    to build the index, then run this file.")
        return
    total_chunks = collection.count()
    if total_chunks == 0:
        print(f"[-] Collection '{args.collection}' is empty. Run the full benchmark first.")
        return

    papers = list_papers(collection)
    total_papers = len(papers)
    print(f"[+] Collection '{args.collection}': {total_papers} papers, {total_chunks} chunks.")
    for p in papers:
        print(f"    - {p['title']} [{p['theme']}] ({p['chunks']} chunks)")
    if total_papers < 31:
        print(f"[!] Expected 31 papers from the Literature Review folder; "
              f"{total_papers} are indexed. Compare this list with the folder "
              f"to see which paper is missing.")

    print("[*] Loading the embedding model (bge-base, once, shared by both versions)...")
    model = rag.load_embedded_model()

    vectorstore = None
    if "langchain" in versions:
        vectorstore = langchain_rag.get_vectorstore(
            persist_directory=args.chroma_dir,
            collection_name=args.collection,
            embeddings=_shared_langchain_embeddings(model),
        )

    results = benchmark_retrieval(collection, model, vectorstore, rag, langchain_rag,
                                  BENCHMARK_QUERIES, top_k=args.top_k, versions=versions)

    total_elapsed = time.perf_counter() - start_all
    print("\n" + "=" * 70)
    print(" COMPARISON (global ranking — the fair numbers)")
    print("=" * 70)
    print(f"{'Version':<12} {'Top-1':>6} {'Top-3':>6} {'Top-5':>6} {'MRR':>7} {'Covered':>8} {'Glob lat':>9} {'PP lat':>8}")
    for version in versions:
        g, c = results[version]["global"], results[version]["coverage"]
        print(f"{version:<12} {g['top1_acc_pct']:>5.1f}% {g['top3_acc_pct']:>5.1f}% {g['top5_acc_pct']:>5.1f}% "
              f"{g['mrr']:>7.4f} {c['coverage_pct']:>7.1f}% {g['avg_retrieval_lat_ms']:>9.2f} "
              f"{c['avg_retrieval_lat_ms']:>8.2f}")
    print("=" * 70)
    print(f" BENCHMARK COMPLETE (Total Runtime: {total_elapsed:.2f}s — no parsing)")
    print("=" * 70)

    output_data = {
        "benchmark": "retrieval-only",
        "versions": versions,
        "top_k": args.top_k,
        "collection": args.collection,
        "chroma_dir": args.chroma_dir,
        "total_papers": total_papers,
        "total_chunks": total_chunks,
        "total_runtime_sec": round(total_elapsed, 2),
        "papers": papers,
        "ir_metrics": results,
    }
    out_file = Path(args.output)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"[+] Benchmark results saved to: {out_file.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval-only benchmark (original vs LangChain) on the existing collection. No parsing.")
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR,
                        help="Chroma persist directory holding the collection (default: ./upload_pdfs_chroma_db)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help="Collection name (default: literature_review_chroma_db)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Global ranking depth per query (same for both versions)")
    parser.add_argument("--versions", choices=["both", "original", "langchain"], default="both",
                        help="Which pipeline version(s) to test")
    parser.add_argument("--output", default="retrieval_benchmark_results.json",
                        help="Results JSON file")
    args = parser.parse_args()
    run_retrieval_benchmark(args)


if __name__ == "__main__":
    main()
