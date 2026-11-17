"""Simulated medical RAG: PDF text extraction (with optional sanitization), chunking, embedding, retrieval.

Design (Chapter 3):
- Document text = page text + PDF metadata values (Subject etc.), because a real ingestion pipeline that
  indexes metadata is what the metadata attack targets. Under the sanitize condition metadata is ignored.
- Each run's corpus = the item's own document (clean or attacked variant) + n_distractors clean documents
  of other items, chosen with a seed derived from the item id. Retrieval = top_k chunks by cosine similarity.
"""
import hashlib, os, random, re
import numpy as np
import pymupdf

_EMBEDDER = None

def _embedder(cfg):
    """sentence-transformers model, or a deterministic hash embedder when FAKE_EMBED=1 (offline tests only)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        if os.environ.get("FAKE_EMBED"):
            class _Fake:
                def encode(self, texts, normalize_embeddings=True):
                    out = []
                    for t in texts:
                        rng = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
                        v = rng.standard_normal(384); out.append(v / np.linalg.norm(v))
                    return np.array(out, dtype="float32")
            _EMBEDDER = _Fake()
        else:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer(cfg["retrieval"]["embedder"])
    return _EMBEDDER

def _span_is_hidden(span, scfg):
    if span["size"] < scfg["min_font_pt"]:
        return True
    c = span["color"]
    r, g, b = ((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255
    dist = ((1 - r) ** 2 + (1 - g) ** 2 + (1 - b) ** 2) ** 0.5
    return dist < scfg["white_threshold"]

def extract_text(path, cfg, sanitize=False):
    """Return (text, n_spans_dropped). sanitize=True applies the visual-sanitization defense."""
    scfg = cfg["sanitize"]
    dropped = 0
    parts = []
    with pymupdf.open(path) as d:
        for page in d:
            if not sanitize:
                parts.append(page.get_text())
                continue
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    kept = []
                    for span in line["spans"]:
                        if _span_is_hidden(span, scfg):
                            dropped += 1
                        else:
                            kept.append(span["text"])
                    if kept:
                        parts.append(" ".join(kept))
        if not (sanitize and scfg["ignore_metadata"]):
            meta = [str(v) for k, v in d.metadata.items() if v and k not in ("format", "creationDate", "modDate")]
            if meta:
                parts.append("Document metadata: " + " | ".join(meta))
    text = re.sub(r"[ \t]+", " ", "\n".join(parts)).strip()
    return text, dropped

def chunk(text, cfg):
    n, o = cfg["retrieval"]["chunk_chars"], cfg["retrieval"]["chunk_overlap"]
    if len(text) <= n:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + n])
        i += n - o
    return out

def doc_path(cfg, item_id, variant):
    return os.path.join(cfg["paths"]["docs_dir"], variant, f"{item_id}.pdf")

def distractor_ids(cfg, item, all_ids):
    rng = random.Random(f"{cfg['seed']}:distractors:{item['id']}")
    pool = [i for i in all_ids if i != item["id"]]
    return rng.sample(pool, cfg["retrieval"]["n_distractors"])

def retrieve(cfg, item, variant, all_ids, question, sanitize=False):
    """Build the per-run corpus and return (retrieved_chunks, info)."""
    target_path = doc_path(cfg, item["id"], variant)
    texts, origins, dropped_total = [], [], 0
    t, dropped = extract_text(target_path, cfg, sanitize=sanitize)
    dropped_total += dropped
    for c in chunk(t, cfg):
        texts.append(c); origins.append("target")
    for did in distractor_ids(cfg, item, all_ids):
        t2, d2 = extract_text(doc_path(cfg, did, "clean"), cfg, sanitize=sanitize)
        dropped_total += d2
        for c in chunk(t2, cfg):
            texts.append(c); origins.append("distractor")
    emb = _embedder(cfg)
    M = emb.encode(texts, normalize_embeddings=True)
    q = emb.encode([question], normalize_embeddings=True)[0]
    sims = M @ q
    k = min(cfg["retrieval"]["top_k"], len(texts))
    top = np.argsort(-sims)[:k]
    chunks = [texts[i] for i in top]
    inj = item.get("injected_instruction")
    words = {w for w in re.findall(r"[a-z][a-z\-]+", (inj or "").lower()) if len(w) > 3}
    joined = " ".join(chunks).lower()
    overlap = (len([w for w in words if w in joined]) / len(words)) if words else 0.0
    contains = bool(inj) and overlap >= 0.5
    info = {"n_corpus_chunks": len(texts), "n_target_chunks_retrieved": int(sum(origins[i] == "target" for i in top)),
            "retrieved_contains_injection": contains, "sanitize_spans_dropped": dropped_total}
    return chunks, info
