"""Chapter 4 descriptive figures and tables of the benchmark (no model outputs).

Usage: python analyze_benchmark.py  -> results/figures/fig_4_1_categories.png, fig_4_2_attack_types.png,
                                       fig_4_3_lengths.png; results/tables/t4_*.csv
"""
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import SERIES, INK, INK2, GRID, AXIS, SURFACE, style
from bench import load_config, load_items

CAT_LABELS = {"management": "Management", "pharmacology": "Pharmacology", "biostat_ethics": "Biostatistics and ethics",
              "diagnosis": "Diagnosis", "pathophysiology": "Pathophysiology", "basic_science": "Basic science",
              "findings": "Clinical findings", "risk_prognosis": "Risk and prognosis",
              "investigation": "Investigation", "other": "Other"}
HARM_LABELS = {"misinformation": "Misinformation", "data_theft": "Data theft", "fraud": "Fraud",
               "harassment": "Harassment", "substances": "Substances", "dangerous_acts": "Dangerous acts"}


def hbar_style(ax, title, xlabel):
    style(ax, title, "")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.spines["bottom"].set_visible(False); ax.spines["left"].set_visible(True); ax.spines["left"].set_color(AXIS)


def main():
    cfg = load_config(); A = cfg["analysis"]
    fd, td = A["figures_dir"], A["tables_dir"]
    os.makedirs(fd, exist_ok=True); os.makedirs(td, exist_ok=True)
    items = load_items(cfg)
    med = [i for i in items if i["source"] == "MedQA"]
    mpib = [i for i in items if i["source"] == "MPIB"]
    adv = [i for i in items if i["source"] == "AdvBench"]

    # ---- Figure 4.1 categories of genuine questions
    c = Counter(i["category"] for i in med).most_common()
    labs = [CAT_LABELS[k] for k, _ in c][::-1]; vals = [v for _, v in c][::-1]
    pd.DataFrame({"category": labs[::-1], "items": vals[::-1],
                  "share %": [round(100 * v / len(med), 1) for v in vals[::-1]]}).to_csv(f"{td}/t4_medqa_categories.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.3, 3.6), dpi=300); fig.patch.set_facecolor(SURFACE)
    ax.barh(labs, vals, color=SERIES[0], height=0.68)
    for y, v in enumerate(vals):
        ax.text(v + 1, y, f"{v}  ({100 * v / len(med):.0f}%)", va="center", fontsize=8, color=INK2)
    hbar_style(ax, "Genuine questions by category", "Number of MedQA questions (total 300)")
    ax.set_xlim(0, max(vals) * 1.22); ax.tick_params(axis="y", labelsize=8.5); ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(f"{fd}/fig_4_1_categories.png", facecolor=SURFACE); plt.close(fig)

    # ---- Figure 4.2 attack types: MPIB rule x severity, AdvBench harm category
    rules = sorted({i["rule"] for i in mpib}); sevs = sorted({i["severity"] for i in mpib})
    tab = pd.crosstab(pd.Series([i["rule"] for i in mpib], name="rule"), pd.Series([i["severity"] for i in mpib], name="severity"))
    tab.to_csv(f"{td}/t4_mpib_rule_severity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.3), dpi=300, gridspec_kw={"width_ratios": [1, 1]}); fig.patch.set_facecolor(SURFACE)
    ax = axes[0]; bottom = np.zeros(len(rules)); sev_col = {2: "#9ec5f4", 3: SERIES[0], 4: "#104281"}
    for s in sevs:
        v = np.array([tab.loc[r, s] if s in tab.columns else 0 for r in rules])
        ax.bar(rules, v, bottom=bottom, color=sev_col[s], width=0.66, label=f"Severity {s}", edgecolor=SURFACE, linewidth=0.8)
        bottom += v
    for x, t in enumerate(bottom):
        ax.text(x, t + 0.6, f"{int(t)}", ha="center", fontsize=8, color=INK2)
    style(ax, "(a) MPIB by rule", "Items (total 150)")
    ax.set_ylim(0, 38); ax.tick_params(labelsize=8); ax.yaxis.label.set_size(8.5); ax.title.set_size(9.5)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12),
              handlelength=1.0, columnspacing=1.0)
    ax = axes[1]; hc = Counter(i["category"] for i in adv)
    hl = [HARM_LABELS[k] for k in HARM_LABELS][::-1]; hv = [hc[k] for k in HARM_LABELS][::-1]
    ax.barh(hl, hv, color=SERIES[1], height=0.66)
    for y, v in enumerate(hv):
        ax.text(v + 0.5, y, str(v), va="center", fontsize=8, color=INK2)
    hbar_style(ax, "(b) AdvBench by harm", "Items (total 150)")
    ax.set_xlim(0, 30); ax.tick_params(labelsize=8); ax.xaxis.label.set_size(8.5); ax.title.set_size(9.5)
    fig.tight_layout(w_pad=1.6); fig.savefig(f"{fd}/fig_4_2_attack_types.png", facecolor=SURFACE); plt.close(fig)

    # ---- Figure 4.3 / Table 4.3 lengths in words
    wl = lambda xs: np.array([len(x.split()) for x in xs])
    series = [("MedQA questions", wl(i["question"] for i in med), SERIES[0]),
              ("MPIB questions", wl(i["question"] for i in mpib), SERIES[2]),
              ("MPIB injected instructions", wl(i["injected_instruction"] for i in mpib), SERIES[2]),
              ("AdvBench injected instructions", wl(i["injected_instruction"] for i in adv), SERIES[1])]
    rows = [{"text": n, "items": len(v), "minimum": int(v.min()), "first quartile": float(np.percentile(v, 25)),
             "median": float(np.median(v)), "mean": round(float(v.mean()), 1), "third quartile": float(np.percentile(v, 75)),
             "maximum": int(v.max()), "standard deviation": round(float(v.std()), 1)} for n, v, _ in series]
    pd.DataFrame(rows).to_csv(f"{td}/t4_lengths.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.3, 3.0), dpi=300); fig.patch.set_facecolor(SURFACE)
    pos = list(range(len(series)))[::-1]
    bp = ax.boxplot([v for _, v, _ in series], positions=pos, orientation="horizontal", widths=0.55, patch_artist=True,
                    flierprops=dict(marker="o", markersize=2.5, markerfacecolor=INK2, markeredgecolor="none", alpha=0.5),
                    medianprops=dict(color=SURFACE, linewidth=1.6), whiskerprops=dict(color=INK2, linewidth=0.9),
                    capprops=dict(color=INK2, linewidth=0.9), boxprops=dict(linewidth=0))
    for patch, (_, _, col) in zip(bp["boxes"], series):
        patch.set_facecolor(col)
    for p, (n, v, _) in zip(pos, series):
        ax.text(max(v) + 8, p, f"median {np.median(v):g}", va="center", fontsize=7.8, color=INK2)
    ax.set_yticks(pos, [n for n, _, _ in series])
    hbar_style(ax, "Length of questions and injected instructions", "Words")
    ax.set_xlim(0, 400); ax.tick_params(axis="y", labelsize=8.5); ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(f"{fd}/fig_4_3_lengths.png", facecolor=SURFACE); plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False)); print(tab)
    short = sum(1 for i in mpib if len(i["injected_instruction"].split()) <= 5)
    longi = sum(1 for i in mpib if len(i["injected_instruction"].split()) > 100)
    print("MPIB injections <=5 words:", short, " >100 words:", longi)


if __name__ == "__main__":
    main()
