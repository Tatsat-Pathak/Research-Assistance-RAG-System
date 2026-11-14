"""
chunk_extracted.py - turn extract_papers.py output into retrieval chunks.

Usage (function style):
    from chunk_extracted import chunk_papers, save_chunks
    chunks = chunk_papers(documents)          # documents = {paper_title: {heading: [item, ...]}}
    save_chunks(chunks, "chunks.jsonl")       # optional

    # each chunk is a dict: chunk_id, paper_id, paper_title, section, section_path, chunk_type,
    # chunk_index, word_count, text (show/cite this), embed_text (embed this), label (tables only)

Rules
  * text + equation items are packed in reading order into ~TARGET-word chunks (never across a
    section), paragraph-aligned, with the last sentence repeated as overlap inside a section
  * a paragraph above MAX words is split on sentence boundaries
  * each table is its own chunk (caption + markdown + footnote); big tables split by rows with
    the header repeated; each code block likewise
  * each figure caption (if kept) is its own small chunk; '(a) ...' sub-captions fold into it
  * very small sections (< MIN_SECTION words) are merged into the previous chunk of the same
    top-level section, with their heading kept inline, instead of becoming near-empty chunks
  * every chunk carries paper title + full section path in `embed_text`
"""
import json
import re

TARGET, MAX, MIN_TAIL, OVERLAP_SENTS = 300, 400, 60, 1
# Defaults kept for backwards compatibility; chunk_papers(target_words=...)
# overrides them per call. MAX is always target_words + 100.
MIN_SECTION = 60

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[$])")
TOP_LEVEL = {"abstract", "introduction", "background", "related work", "methods", "methodology",
             "results", "discussion", "conclusion", "conclusions", "references"}
TOP_LEVEL_RE = re.compile(r"^(appendix|acknowledg|declaration|conflicts? of interest|funding|data availability|"
                          r"author contributions?|supplementary|multimedia appendix|credit)", re.I)
END_HEADING = re.compile(r"^\W*(\d+\.?\s*)?(references|bibliography|literature cited)\W*$", re.I)
ABSTRACT_RE = re.compile(r"\babstract\b", re.I)
SUBCAPTION = re.compile(r"^\(?[a-z]\)\s")
LABEL = re.compile(r"^\W*((?i:table|fig(?:ure)?))\.?\s*(\d+|[IVXLC]+)\b")


# ---------- hierarchy ----------
def heading_depth(h, prev_depth, prev_empty):
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s", h)
    if m:
        return len(m.group(1).split("."))
    if h.strip().rstrip(":").lower() in TOP_LEVEL or TOP_LEVEL_RE.match(h.strip()):
        return 1
    if prev_depth is None:
        return 1
    return prev_depth + 1 if prev_empty else prev_depth


# ---------- packing ----------
def split_long(par, target=TARGET):
    sents, out, cur, n = SENT_SPLIT.split(par), [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > target:
            out.append(" ".join(cur)); cur, n = [], 0
        cur.append(s); n += w
    if cur:
        out.append(" ".join(cur))
    return out


def pack(paragraphs, target=TARGET, max_words=MAX):
    units = []
    for p in paragraphs:
        units.extend(split_long(p, target) if len(p.split()) > max_words else [p])
    chunks, cur, n = [], [], 0                      # members: (text, is_overlap)
    for u in units:
        w = len(u.split())
        if cur and n + w > target:
            chunks.append(cur)
            tail = SENT_SPLIT.split(cur[-1][0])[-OVERLAP_SENTS:] if OVERLAP_SENTS else []
            cur = [(" ".join(tail), True)] if tail else []
            n = sum(len(m[0].split()) for m in cur)
        cur.append((u, False)); n += w
    if cur:
        chunks.append(cur)
    if len(chunks) > 1 and sum(len(m[0].split()) for m in chunks[-1] if not m[1]) < MIN_TAIL:
        chunks[-2] += [m for m in chunks[-1] if not m[1]]
        chunks.pop()
    return ["\n\n".join(m[0] for m in c) for c in chunks]


def split_rows(prefix, header, rows, suffix, target=TARGET):
    pieces, cur, n = [], [], 0
    for r in rows:
        w = len(r.split())
        if cur and n + w > target:
            pieces.append(cur); cur, n = [], 0
        cur.append(r); n += w
    pieces.append(cur)
    tot = len(pieces)
    return ["\n\n".join(x for x in ((prefix + (f" (part {i}/{tot})" if tot > 1 else "")).strip(),
                                    "\n".join(header + part), suffix) if x)
            for i, part in enumerate(pieces, 1)]


def table_pieces(it, target=TARGET, max_words=MAX):
    lines = it["content"].split("\n")
    if len(it["content"].split()) <= max_words or len(lines) <= 3:
        return ["\n\n".join(x for x in (it["caption"], it["content"], it.get("footnote", "")) if x)]
    return split_rows(it["caption"], lines[:2], lines[2:], it.get("footnote", ""), target)


def code_pieces(it, target=TARGET, max_words=MAX):
    lines = it["content"].split("\n")
    if len(it["content"].split()) <= max_words or len(lines) <= 3:
        return ["\n\n".join(x for x in (it["caption"], it["content"]) if x)]
    return split_rows(it["caption"], [], lines, "", target)


def fold_subcaptions(items):
    """'(a) Baseline ...' captions are appended to the nearest labelled figure caption."""
    figs = [i for i in items if i["type"] == "figure"]
    for f in list(figs):
        if SUBCAPTION.match(f["content"]) and len(f["content"].split()) < 12:
            k = figs.index(f)
            host = next((g for g in reversed(figs[:k]) if LABEL.match(g["content"])), None) \
                or next((g for g in figs[k + 1:] if LABEL.match(g["content"])), None)
            if host:
                host["content"] += " " + f["content"]
                host["caption"] = host["content"]
                items.remove(f); figs.remove(f)


# ---------- build ----------
def chunk_papers(documents, target_words: int = TARGET):
    """documents: {paper_title: {heading: [item, ...]}} -> list of chunk dicts.

    target_words is the target size of a prose chunk, in words (default 300).
    A single paragraph longer than target_words + 100 is split on sentences.
    """
    max_words = target_words + 100
    out = []
    for p_idx, (title, sections) in enumerate(documents.items()):
        pid, c = f"p{p_idx:02d}", 0
        stack, prev_depth, prev_empty = [], None, False
        paper_chunks = []

        def emit(h, path, ctype, text, label=None, mergeable=False):
            nonlocal c
            safe_title = re.sub(r"[^\w\-.]", "_", title).strip("_")
            safe_h = re.sub(r"[^\w\-.]", "_", h).strip("_")
            rec = {"chunk_id": f"{safe_title}.pdf_{safe_h}_{c}_chunk_{c}", "paper_id": pid, "paper_title": title, "section": h,
                   "section_path": path, "chunk_type": ctype, "chunk_index": c,
                   "word_count": len(text.split()), "text": text,
                   "embed_text": f"Paper: {title}\nSection: {' > '.join(path)}\n\n{text}",
                   "_mergeable": mergeable}
            if label:
                rec["label"] = label
            paper_chunks.append(rec); c += 1

        for h, items in sections.items():
            depth = heading_depth(h, prev_depth, prev_empty)
            while stack and stack[-1][0] >= depth:
                stack.pop()
            stack.append((depth, h))
            prev_depth, prev_empty = depth, (not items) and not END_HEADING.match(h)
            path = [x for _, x in stack]
            fold_subcaptions(items)

            prose = [i["content"] for i in items if i["type"] in ("text", "equation")]
            prose_words = sum(len(p.split()) for p in prose)
            ptype = "abstract" if ABSTRACT_RE.search(h) else "body"

            # tiny section -> merge into the previous prose chunk of the same top-level section
            prev = next((x for x in reversed(paper_chunks) if x["chunk_type"] in ("body", "abstract")), None)
            if (prose and prose_words < MIN_SECTION and ptype == "body" and prev
                    and prev["section_path"][0] == path[0]
                    and prev["word_count"] + prose_words + 3 <= max_words):
                add = h + "\n" + "\n\n".join(prose)
                prev["text"] += "\n\n" + add
                prev["word_count"] = len(prev["text"].split())
                prev["embed_text"] = f"Paper: {title}\nSection: {' > '.join(prev['section_path'])}\n\n{prev['text']}"
            else:
                for text in pack(prose, target_words, max_words):
                    emit(h, path, ptype, text)

            for i in items:
                if i["type"] == "table":
                    for piece in table_pieces(i, target_words, max_words):
                        emit(h, path, "table", piece, i.get("label"))
                elif i["type"] == "figure":
                    emit(h, path, "figure_caption", i["content"], i.get("label"))
                elif i["type"] == "code":
                    for piece in code_pieces(i, target_words, max_words):
                        emit(h, path, "code", piece)
        for rec in paper_chunks:
            rec.pop("_mergeable")
        out.extend(paper_chunks)
    return out


def save_chunks(chunks, path):
    """Write chunks as JSON Lines (one chunk per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")