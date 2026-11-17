"""Chapter 4 diagnostics of the two defenses that change the text, measured without any model.

Sanitization: what is dropped from each document variant and whether the injected text survives.
MAP-style filter: what is removed from the retrieved passages of white-text attacks and of clean genuine documents.
Usage: python analyze_defenses.py -> results/tables/t4_sanitize_diag.csv, t4_mapfilter_diag.csv
"""
import re
from collections import Counter, defaultdict

import pandas as pd

from bench import load_config, load_items, read_jsonl
from defenses import map_filter
from precompute_retrieval import injection_overlap
from rag import extract_text, doc_path

STAGES = {"stage1_regex": "stage 2 (phrases)", "stage2_imperative": "stage 3 (imperative)", "stage3_shape": "stage 4 (shape)"}


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def main():
    cfg = load_config(); td = cfg["analysis"]["tables_dir"]; prim = cfg["analysis"]["primary_technique"]
    items = {i["id"]: i for i in load_items(cfg)}
    R = list(read_jsonl(cfg["paths"]["retrieval_cache"]))

    san = defaultdict(Counter)
    for r in R:
        if not r["sanitize"]:
            continue
        it = items[r["item_id"]]; inj = it.get("injected_instruction"); c = san[r["variant"]]
        text, dropped = extract_text(doc_path(cfg, it["id"], r["variant"]), cfg, sanitize=True)
        clean, _ = extract_text(doc_path(cfg, it["id"], "clean"), cfg, sanitize=True)
        c["documents"] += 1; c["with spans dropped"] += dropped > 0; c["text equal to clean document"] += text == clean
        if inj:
            c["injected text in sanitized document"] += norm(inj) in norm(text)
            c["injected text in retrieved passages"] += norm(inj) in norm(" ".join(r["chunks"]))
            c["overlap flag still true"] += r["retrieved_contains_injection"]
    t = pd.DataFrame(san).T; t.to_csv(f"{td}/t4_sanitize_diag.csv"); print(t.to_string())

    mf = defaultdict(Counter)
    for r in R:
        if r["sanitize"]:
            continue
        it = items[r["item_id"]]
        if r["variant"] == prim and r["retrieved_contains_injection"]:
            c = mf[it["source"]]; outs = [map_filter(ch) for ch in r["chunks"]]
            after = injection_overlap(it["injected_instruction"], [o[0] for o in outs])
            c["items"] += 1; c["a sentence removed"] += any(o[1] for o in outs)
            c["injection below half"] += after < 0.5; c["injection unchanged"] += after >= r["injection_overlap"] - 1e-9
            for o in outs:
                for k, v in o[2].items():
                    if k in STAGES: c[STAGES[k]] += v
        elif r["variant"] == "clean":
            c = mf["clean genuine"]
            for ch in r["chunks"]:
                _, removed, counts = map_filter(ch)
                c["chunks"] += 1; c["chunks changed"] += bool(removed); c["sentences removed"] += len(removed)
                for k, v in counts.items():
                    if k in STAGES: c[STAGES[k]] += v
    t = pd.DataFrame(mf).T.fillna(0).astype(int); t.to_csv(f"{td}/t4_mapfilter_diag.csv"); print(t.to_string())


if __name__ == "__main__":
    main()
