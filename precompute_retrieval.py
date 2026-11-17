"""Precompute retrieval ONCE for every (item, variant, sanitize) the experiment needs -> data/retrieval.jsonl.

Why: retrieval does not depend on the model under test, and loading the embedding model (torch) next to
Ollama on a 16 GB Mac pushes the LLM into swap. run_experiment.py reads this cache and never imports torch.

Also prints retrieval diagnostics (Chapter 4, "Retrieval behaviour"):
  - share of runs where the item's own document contributes >= 1 of the top-k chunks
  - share of attack runs where the injected text is among the retrieved chunks
  - mean similarity of the best target chunk vs the best distractor chunk

Usage: python precompute_retrieval.py            (resume-safe; ~2,400 retrievals)
       python precompute_retrieval.py --limit 60 --diag-only   (quick look)
"""
import argparse, json, os, re
import numpy as np
from bench import load_config, load_items
from rag import extract_text, chunk, doc_path, distractor_ids, _embedder

def content_words(s):
    return {w for w in re.findall(r"[a-z][a-z\-]+", (s or "").lower()) if len(w) > 3}

def injection_overlap(inj, chunks):
    words = content_words(inj)
    if not words:
        return 0.0
    text = " ".join(chunks).lower()
    return len([w for w in words if w in text]) / len(words)

def needed(cfg, items):
    """(item, variant, sanitize) triples the experiment will ask for."""
    techs = set()
    for c in cfg["conditions"]:
        techs |= set(cfg["techniques_by_condition"].get(c, cfg["techniques_by_condition"]["default"]))
    out = []
    for it in items:
        if it["type"] == "genuine":
            out += [(it, "clean", False), (it, "clean", True)]
        else:
            for t in sorted(techs):
                out.append((it, t, False))
            for t in cfg["techniques_by_condition"]["sanitize"]:
                out.append((it, t, True))
    return out

def key(item_id, variant, sanitize):
    return f"{item_id}|{variant}|{'san' if sanitize else 'raw'}"

def retrieve_diag(cfg, item, variant, all_ids, sanitize, emb):
    texts, origins, dropped_total = [], [], 0
    t, d = extract_text(doc_path(cfg, item["id"], variant), cfg, sanitize=sanitize); dropped_total += d
    for c in chunk(t, cfg):
        texts.append(c); origins.append("target")
    for did in distractor_ids(cfg, item, all_ids):
        t2, d2 = extract_text(doc_path(cfg, did, "clean"), cfg, sanitize=sanitize); dropped_total += d2
        for c in chunk(t2, cfg):
            texts.append(c); origins.append("distractor")
    M = emb.encode(texts, normalize_embeddings=True)
    q = emb.encode([item["question"]], normalize_embeddings=True)[0]
    sims = M @ q
    k = min(cfg["retrieval"]["top_k"], len(texts))
    top = [int(i) for i in np.argsort(-sims)[:k]]
    tgt = [float(sims[i]) for i in range(len(texts)) if origins[i] == "target"]
    dis = [float(sims[i]) for i in range(len(texts)) if origins[i] == "distractor"]
    inj = item.get("injected_instruction")
    chunks = [texts[i] for i in top]
    return {
        "key": key(item["id"], variant, sanitize), "item_id": item["id"], "variant": variant, "sanitize": sanitize,
        "chunks": chunks, "n_corpus_chunks": len(texts),
        "n_target_chunks_retrieved": sum(origins[i] == "target" for i in top),
        "best_target_sim": round(max(tgt), 3) if tgt else None,
        "best_distractor_sim": round(max(dis), 3) if dis else None,
        "injection_overlap": round(injection_overlap(inj, chunks), 3) if inj else None,
        "retrieved_contains_injection": bool(inj) and injection_overlap(inj, chunks) >= 0.5,
        "sanitize_spans_dropped": dropped_total,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--diag-only", action="store_true", help="print diagnostics from the existing cache and exit")
    a = ap.parse_args()
    cfg = load_config()
    out_path = cfg["paths"].get("retrieval_cache", "data/retrieval.jsonl")
    rows = {}
    if os.path.exists(out_path):
        for l in open(out_path):
            if l.strip():
                r = json.loads(l); rows[r["key"]] = r
    if not a.diag_only:
        items = load_items(cfg, require_advbench=True)
        all_ids = [it["id"] for it in items]
        todo = [(it, v, s) for it, v, s in needed(cfg, items) if key(it["id"], v, s) not in rows]
        if a.limit:
            todo = todo[:a.limit]
        print(f"[retrieval] {len(rows)} cached, {len(todo)} to compute -> {out_path}")
        emb = _embedder(cfg)
        with open(out_path, "a") as f:
            for n, (it, v, s) in enumerate(todo, 1):
                r = retrieve_diag(cfg, it, v, all_ids, s, emb)
                f.write(json.dumps(r) + "\n"); rows[r["key"]] = r
                if n % 100 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)}", flush=True)
    # diagnostics
    R = list(rows.values())
    if not R:
        print("nothing cached yet"); return
    def share(rs, f): return f"{100 * sum(f(r) for r in rs) / max(1, len(rs)):.0f}%"
    gen = [r for r in R if r["item_id"].startswith("MEDQA")]
    att = [r for r in R if not r["item_id"].startswith("MEDQA")]
    raw_att = [r for r in att if not r["sanitize"]]
    san_att = [r for r in att if r["sanitize"]]
    print("\n=== retrieval diagnostics ===")
    print(f"rows: {len(R)}  (genuine {len(gen)}, attack {len(att)})")
    print("own document in top-k   genuine:", share(gen, lambda r: r["n_target_chunks_retrieved"] > 0),
          "| attack:", share(att, lambda r: r["n_target_chunks_retrieved"] > 0))
    print("injection retrieved     no sanitize:", share(raw_att, lambda r: r["retrieved_contains_injection"]),
          "| with sanitize:", share(san_att, lambda r: r["retrieved_contains_injection"]))
    bt = [r["best_target_sim"] for r in R if r["best_target_sim"] is not None]
    bd = [r["best_distractor_sim"] for r in R if r["best_distractor_sim"] is not None]
    print(f"mean best similarity    target {np.mean(bt):.3f} | distractor {np.mean(bd):.3f}")
    worst = sorted(gen, key=lambda r: (r["best_target_sim"] or 0) - (r["best_distractor_sim"] or 0))[:5]
    for r in worst:
        print(f"  weakest genuine: {r['item_id']} target {r['best_target_sim']} vs distractor {r['best_distractor_sim']}")

if __name__ == "__main__":
    main()
