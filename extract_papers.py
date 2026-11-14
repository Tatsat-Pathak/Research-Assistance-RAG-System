"""
extract_papers.py - content extraction from MinerU output, in the same spirit as your loop
(abstract -> conclusion / before references) but:
  * no bbox-area filtering: front matter is detected by position (text before the first section
    heading), not by size, so short paragraphs, list items and equation definitions are kept
  * handles text, equation, table, image/chart, code (your version only handled text)
  * rejoins paragraphs split across columns/pages (continues_prev) and tables split across pages
  * a repeated doc_title (ACM cover page) restarts the paper instead of keeping the cover
  * tables/figures are filed under the section that discusses them most, not the section that
    happens to precede them in reading order

KEEP_TYPES (below) decides which of equation/table/figure/code are kept; default is text, equation, table and code (no images / figure captions).

Output per paper:  sections = {heading: [item, ...]}  where item is one of
  {"type": "text",     "content": str}
  {"type": "equation", "content": "$$...$$"}
  {"type": "table",    "caption": str, "content": markdown, "footnote": str, "label": "Table 2"|None}
  {"type": "figure",   "caption": str, "content": str(=caption), "label": "Figure 1"|None}
  {"type": "code",     "caption": str, "content": str}
"""
import re

# Which item types to keep. Images and figure captions are off; add "figure" to bring back
# image/chart CAPTIONS (the image data itself is never kept).
KEEP_TYPES = {"text", "equation", "table", "code"}
STOP_AT_CONCLUSION = True            # keep the conclusion, then stop at acknowledgments/appendix/etc.
KEEP_FLOATS_AFTER_REFERENCES = True  # appendix tables/figures after the reference list -> "Appendix"
MIN_ABSTRACT_WORDS = 60              # unlabeled front-matter block this long = the abstract

SKIP = {"header", "footer", "page_number", "page_footnote", "aside_text"}
NOISE_HEADING = re.compile(r"^(journal of .*|proceedings of .*|arxiv.*|preprint.*|page \d+.*|\d+)$", re.I)
ABSTRACT_RE = re.compile(r"\babstract\b", re.I)
META_HEAD = re.compile(r"^\W*((acm )?reference format|cite as|index terms|keywords?|general terms|"
                       r"categories and subject descriptors|ccs concepts|article ?info|jel classification)\W*$", re.I)
_NUM = r"(?:\d+\.?\s*|[IVXivx]+\.\s*)?"
REFS = re.compile(rf"^\W*{_NUM}(references|bibliography|literature cited)\W*$", re.I)
TAIL = re.compile(rf"^\W*{_NUM}(acknowledg|appendix|declaration|conflicts? of interest|funding|data availability|"
                  r"author contributions?|credit|supplementary|multimedia appendix|abbreviations)", re.I)
CONCL = re.compile(r"\bconclu[a-z]*\b", re.I)
NOT_CONCL = re.compile(r"validity|threat", re.I)
LABEL = re.compile(r"^\W*((?i:table|fig(?:ure)?))\.?\s*(\d+|[IVXLC]+)\b")


# ---------- small helpers ----------
def clean(text):
    text = re.sub(r"</?(strong|u|em|i|b)>", "", text)
    text = re.sub(r"<sup>(.*?)</sup>", r"^\1", text)
    text = re.sub(r"<sub>(.*?)</sub>", r"_\1", text)
    text = re.sub(r"<[^>]+>", "", text).replace("**", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fix_spaced(h):
    """'A B S T R A C T' -> 'ABSTRACT' (runs of >=4 single letters only, so 'Type A or B' is safe)."""
    return re.sub(r"\b(?:[A-Za-z] ){3,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), h)


def _take_pending(pending, out, col):
    """Fill cells that are covered by a rowspan from a row above."""
    while col in pending:
        txt, left = pending[col]
        out.append(txt)
        if left <= 1:
            del pending[col]
        else:
            pending[col][1] = left - 1
        col += 1
    return col


def html_table_to_md(html):
    """HTML table -> markdown, expanding colspan (empty padding cells) and rowspan (value repeated
    down), so every row has the same number of cells."""
    grid, pending = [], {}
    for r in re.findall(r"<tr.*?>(.*?)</tr>", html, re.S | re.I):
        out, col = [], 0
        for _, attrs, inner in re.findall(r"<t([dh])([^>]*)>(.*?)</t[dh]>", r, re.S | re.I):
            col = _take_pending(pending, out, col)
            txt = clean(inner).replace("\n", " ")
            cs = re.search(r"colspan=\"?(\d+)", attrs, re.I)
            rs = re.search(r"rowspan=\"?(\d+)", attrs, re.I)
            cs, rs = int(cs.group(1)) if cs else 1, int(rs.group(1)) if rs else 1
            for k in range(cs):
                out.append(txt if k == 0 else "")
                if rs > 1:
                    pending[col] = [txt if k == 0 else "", rs - 1]
                col += 1
        _take_pending(pending, out, col)
        grid.append(out)
    width = max((len(r) for r in grid), default=0)
    md = []
    for i, r in enumerate(grid):
        md.append("| " + " | ".join(r + [""] * (width - len(r))) + " |")
        if i == 0:
            md.append("| " + " | ".join("---" for _ in range(width)) + " |")
    return "\n".join(md)


def blocks_from_structured(structured):
    """doc.structured_content() -> flat list of blocks (page order kept)."""
    return [b for p in structured["pages"] for b in p["blocks"]]


def item_text(item):
    """Plain text of an item, for embedding / chunking."""
    if item["type"] == "table":
        return "\n\n".join(x for x in (item["caption"], item["content"], item.get("footnote", "")) if x)
    if item["type"] == "code" and item.get("caption"):
        return item["caption"] + "\n\n" + item["content"]
    return item["content"]


def _kind(block_type):
    """MinerU block type -> item type used in KEEP_TYPES."""
    return {"image": "figure", "chart": "figure"}.get(block_type, block_type)


def _float_item(b):
    """Build a table / figure / code item from a block (None if there is nothing worth keeping)."""
    t, c = b["type"], (b.get("content") or "")
    caps = [clean(x.get("content", "")) for x in b.get("captions", []) if clean(x.get("content", ""))]
    if t == "table":
        body = html_table_to_md(c) if "<table" in c.lower() else "\n".join(clean(l) for l in c.strip().split("\n"))
        foot = " ".join(clean(x.get("content", "")) for x in b.get("footnotes", []))
        if not (body or caps):
            return None
        cap = " ".join(caps)
        if "figure" not in KEEP_TYPES and re.search(r"\bfig(?:ure)?\.?\s*\d+", cap, re.I) \
                and not re.search(r"\btable\s*(\d+|[IVXLC]+)\b", cap, re.I):
            return None                                    # a chart MinerU mislabelled as a table
        return {"type": "table", "caption": " ".join(caps), "content": body, "footnote": foot}
    if t == "code":
        if not c.strip():
            return None
        return {"type": "code", "caption": " ".join(caps), "content": c.strip()}
    return None


# ---------- main extraction ----------
def extract_paper(blocks, fallback_title="untitled"):
    title = tnorm = None
    sections, meta_sections, front, refs = {}, {}, [], []
    cur_items = None            # list that receives content right now
    started = done = seen_concl = False

    def new_section(h):
        key, k = h, 2
        while key in sections:
            key, k = f"{h} ({k})", k + 1
        sections[key] = []
        return sections[key]

    def add_float(item_blocks_type, b):
        if item_blocks_type in ("image", "chart"):
            return [{"type": "figure", "caption": cap, "content": cap}
                    for cap in (clean(x.get("content", "")) for x in b.get("captions", [])) if cap]
        it = _float_item(b)
        return [it] if it else []

    for b in blocks:
        t, c = b["type"], b.get("content") or ""

        if t == "doc_title":
            n = norm(c)
            if title is not None and n == tnorm:          # repeated title -> what we had was a cover page
                sections, meta_sections, front = {}, {}, []
                cur_items, started, done, seen_concl = None, False, False, False
            title, tnorm = clean(c), n
            continue
        if title is None:                                 # cover / journal header before the title
            continue

        if t == "ref_text":
            if clean(c):
                if b.get("continues_prev") and refs:
                    refs[-1] += " " + clean(c)
                else:
                    refs.append(clean(c))
            continue
        if t in SKIP:
            continue

        if done:                                          # after References / after the conclusion tail
            if KEEP_FLOATS_AFTER_REFERENCES and t in ("table", "image", "chart", "code") and _kind(t) in KEEP_TYPES:
                items = add_float(t, b)
                if items:
                    sections.setdefault("Appendix", []).extend(items)
            continue

        if t == "paragraph_title":
            h = fix_spaced(clean(c))
            if NOISE_HEADING.match(h) and cur_items is not None:
                continue
            if REFS.match(h) or (STOP_AT_CONCLUSION and seen_concl and TAIL.match(h)):
                done = True
                continue
            if META_HEAD.match(h):                        # keywords etc. -> metadata, not content
                cur_items = meta_sections.setdefault(h, [])
                continue
            started = True
            cur_items = new_section(h)
            if STOP_AT_CONCLUSION and CONCL.search(h) and not NOT_CONCL.search(h):
                seen_concl = True

        elif t == "text":
            txt = clean(c)
            if not txt:
                continue
            if not started:                               # authors, affiliations, or an unlabeled abstract
                front.append(txt)
                continue
            last = next((x for x in reversed(cur_items) if x["type"] == "text"), None)
            if b.get("continues_prev") and last:
                last["content"] += " " + txt              # paragraph split across column/page
            else:
                cur_items.append({"type": "text", "content": txt})

        elif t == "equation":
            if started and c.strip() and "equation" in KEEP_TYPES:
                cur_items.append({"type": "equation", "content": f"$${c.strip()}$$"})

        elif t in ("table", "image", "chart", "code") and started and _kind(t) in KEEP_TYPES:
            if t == "table" and b.get("continues_prev"):
                prev = next((x for x in reversed(cur_items) if x["type"] == "table"), None)
                if prev and not b.get("captions"):
                    rows = (_float_item(b) or {"content": ""})["content"].split("\n")
                    if rows[:1] == prev["content"].split("\n")[:1]:
                        rows = rows[2:]                   # drop the repeated header row
                    prev["content"] += "\n" + "\n".join(rows)
                    continue
            cur_items.extend(add_float(t, b))

    # abstract printed without a heading: promote long front-matter blocks
    long_front = [x for x in front if len(x.split()) >= MIN_ABSTRACT_WORDS]
    if long_front and not any(ABSTRACT_RE.search(h) for h in sections):
        sections = {"Abstract": [{"type": "text", "content": x} for x in long_front], **sections}
        front = [x for x in front if x not in long_front]

    _assign_floats(sections)
    extras = {"front_matter": front, "meta_sections": {k: [i["content"] for i in v if i["type"] == "text"]
                                                       for k, v in meta_sections.items()},
              "references": refs}
    return title or fallback_title, sections, extras


def _assign_floats(sections):
    """Tables/figures float to a page top, so reading order files them under the previous section.
    Move each to the section whose prose mentions its label most ('see Table 3' in an earlier
    section is a forward reference and should not win)."""
    names = list(sections)
    moved = {n: [] for n in names}
    for n in names:
        keep = []
        for it in sections[n]:
            if it["type"] not in ("table", "figure"):
                keep.append(it); continue
            m = LABEL.match(it["caption"] or "")
            it["label"] = f"{m.group(1).title()} {m.group(2)}" if m else None
            target = n
            if m:
                kind = "table" if m.group(1).lower() == "table" else r"fig(?:ure)?\.?"
                pat = re.compile(rf"\b(?i:{kind})\s*{m.group(2)}\b")
                counts = {k: sum(len(pat.findall(x["content"])) for x in v if x["type"] == "text")
                          for k, v in sections.items()}
                best = max(counts, key=counts.get)
                if counts[best] > 0:
                    target = best
            if target == n:
                keep.append(it)
            else:
                moved[target].append(it)
        sections[n] = keep
    for n in names:
        sections[n].extend(moved[n])