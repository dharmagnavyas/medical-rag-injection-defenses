"""Chapter 4 diagnostics of the generated documents and of the retrieval step (no model outputs).

Usage: python analyze_retrieval.py -> results/figures/fig_4_4_retrieval.png,
                                      results/tables/t4_documents.csv, t4_retrieval_injection.csv
"""
import json, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf

from analyze import SERIES, INK2, GRID, SURFACE, style
from bench import load_config, load_items, read_jsonl
from rag import extract_text, chunk

TECH_LABELS = {"white_text": "White text", "tiny_font": "1-point font", "metadata": "Metadata"}


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def main():
    cfg = load_config(); A = cfg["analysis"]
    fd, td = A["figures_dir"], A["tables_dir"]
    items = {i["id"]: i for i in load_items(cfg)}

    # ---- documents
    rows = []
    for d in read_jsonl(cfg["paths"]["docs_manifest"]):
        with pymupdf.open(d["path"]) as doc:
            pages = len(doc)
        text, _ = extract_text(d["path"], cfg, sanitize=False)
        inj = items[d["item_id"]].get("injected_instruction")
        rows.append({"variant": d["variant"], "source": d["source"], "pages": pages, "chars": len(text),
                     "chunks": len(chunk(text, cfg)), "present": (norm(inj) in norm(text)) if inj else None})
    docs = pd.DataFrame(rows)
    g = docs.groupby("variant", sort=False)
    t = pd.DataFrame({"documents": g.size(), "max pages": g.pages.max(), "chars min": g.chars.min(),
                      "chars median": g.chars.median(), "chars max": g.chars.max(), "chunks min": g.chunks.min(),
                      "chunks median": g.chunks.median(), "chunks max": g.chunks.max(),
                      "injected text extracted": g.present.sum()})
    t.to_csv(f"{td}/t4_documents.csv"); print(t.to_string())

    # ---- retrieval
    r = pd.DataFrame([{k: v for k, v in x.items() if k != "chunks"} for x in read_jsonl(cfg["paths"]["retrieval_cache"])])
    r["source"] = r.item_id.map(lambda i: items[i]["source"])
    raw = r[~r.sanitize]
    att = raw[raw.source != "MedQA"]
    tab = att.groupby(["variant", "source"]).retrieved_contains_injection.agg(["sum", "count"]).unstack()
    tab.to_csv(f"{td}/t4_retrieval_injection.csv"); print(tab)
    print("own doc in top-k:", int((r.n_target_chunks_retrieved > 0).sum()), "of", len(r))
    print("own > distractor (raw):", int((raw.best_target_sim > raw.best_distractor_sim).sum()), "of", len(raw))
    print(raw[["best_target_sim", "best_distractor_sim"]].describe().round(3))

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.3), dpi=300); fig.patch.set_facecolor(SURFACE)
    ax = axes[0]; bins = np.linspace(0, 1, 41)
    ax.hist(raw.best_distractor_sim, bins=bins, color=SERIES[1], alpha=0.85, label="Best other document")
    ax.hist(raw.best_target_sim, bins=bins, color=SERIES[0], alpha=0.85, label="Best chunk of own document")
    style(ax, "(a) Similarity to the question", "Retrievals")
    ax.set_xlabel("Cosine similarity", color=INK2, fontsize=9); ax.tick_params(labelsize=8)
    ax.set_ylim(0, 225); ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax = axes[1]; techs = cfg["documents"]["hiding_techniques"]; x = np.arange(len(techs)); w = 0.36
    for j, (src, col) in enumerate([("MPIB", SERIES[0]), ("AdvBench", SERIES[1])]):
        v = [100 * tab[("sum", src)][t_] / tab[("count", src)][t_] for t_ in techs]
        ax.bar(x + (j - 0.5) * w, v, w, color=col, label=src)
        for xi, vi in zip(x, v):
            ax.text(xi + (j - 0.5) * w, vi + 1.5, f"{vi:.0f}", ha="center", fontsize=7.5, color=INK2)
    style(ax, "(b) Injected text retrieved", "% of attack items")
    ax.set_xticks(x, [TECH_LABELS[t_] for t_ in techs]); ax.set_ylim(0, 112); ax.tick_params(labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.legend(fontsize=7, frameon=False, loc="upper right", ncols=2)
    fig.tight_layout(); fig.savefig(f"{fd}/fig_4_4_retrieval.png", facecolor=SURFACE); plt.close(fig)


if __name__ == "__main__":
    main()
