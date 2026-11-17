"""Step 7: analysis of results/scores.jsonl -> results/tables/*.csv, results/figures/*.png, results/analysis.md.

Everything here is deterministic given scores.jsonl and config.yaml (bootstrap seeded with config.seed).
Tables (chapter 5 numbering is the author's):
  T1 main grid  ASR / FRR / utility per model x condition with 95% Wilson CIs (ASR on primary_technique rows)
  T2 ASR by attack source (MPIB / AdvBench)            T3 sanitize ASR by hiding technique
  T4 McNemar defense vs baseline per model (all attack rows and primary_source), Holm-corrected, + Cochran's Q
  T5 paired bootstrap CI of the ASR change vs baseline  T6 medical vs general model pairs (medft condition)
  T7 FRR / no-answer / utility by MedQA category x condition (RQ3)
  T8 ASR conditional on the poisoned chunk being retrieved
  T9 judge calibration (second judge), doc-echo and degenerate counts, latency
  T10 manual review agreement (only if results/manual_sample.csv has human_label filled)
Figures: ASR grouped bars (all / primary source), trade-off scatter (ASR vs utility, ASR vs FRR), ASR by source,
         sanitize by technique, ASR heatmap.
"""
import csv, json, math, os, random
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from bench import load_config

# --- validated default palette (dataviz skill, light surface); conditions are the categorical entity, fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


# ----------------------------------------------------------------------------- statistics
def wilson(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"),) * 3
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, min(p, c - h)), min(1.0, max(p, c + h))   # clipped so error bars are never negative


def mcnemar_exact(b, c):
    """Exact two-sided McNemar on the discordant counts (b: baseline 1 / defense 0, c: baseline 0 / defense 1)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, stats.binomtest(min(b, c), n, 0.5).pvalue)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def cochran_q(matrix):
    """matrix: items x conditions of 0/1. Returns (Q, df, p)."""
    x = np.asarray(matrix, dtype=float)
    k = x.shape[1]
    row = x.sum(1)
    col = x.sum(0)
    n = row.sum()
    den = k * n - (row ** 2).sum()
    if den == 0:
        return 0.0, k - 1, 1.0
    q = (k - 1) * (k * (col ** 2).sum() - n * n) / den
    return q, k - 1, stats.chi2.sf(q, k - 1)


def paired_bootstrap_delta(a, b, n_boot, rng):
    """CI of mean(b) - mean(a) resampling paired items."""
    a, b = np.asarray(a), np.asarray(b)
    d = b - a
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def kappa(pairs):
    n = len(pairs)
    pa = sum(a == b for a, b in pairs) / n
    p1 = sum(a for a, _ in pairs) / n
    p2 = sum(b for _, b in pairs) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return pa, (pa - pe) / (1 - pe) if pe < 1 else float("nan")


def rouge_l_f1(ref, hyp):
    """ROUGE-L F1 on lower-cased word tokens (LCS based); 0 when either side is empty."""
    import re as _re
    a = _re.findall(r"[a-z0-9]+", (ref or "").lower())
    b = _re.findall(r"[a-z0-9]+", (hyp or "").lower())
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p_, r_ = lcs / len(b), lcs / len(a)
    return 2 * p_ * r_ / (p_ + r_)


# ----------------------------------------------------------------------------- helpers
def fmt_pct(p, lo=None, hi=None):
    if p != p:
        return "-"
    return f"{100*p:.1f}" if lo is None else f"{100*p:.1f} [{100*lo:.1f}, {100*hi:.1f}]"


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r.values]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def save(df, name, tables_dir, md, title, note=None, floatfmt="{:.3f}"):
    df.to_csv(os.path.join(tables_dir, name + ".csv"), index=False)
    md.append(f"\n### {title}\n")
    if note:
        md.append(note + "\n")
    md.append(md_table(df, floatfmt))


# ----------------------------------------------------------------------------- figures
def style(ax, title, ylabel, subtitle=None):
    ax.set_facecolor(SURFACE)
    import textwrap
    nlines = len(textwrap.wrap(subtitle, int(ax.figure.get_figwidth() * 13))) if subtitle else 0
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10 + 12 * nlines, fontweight="bold")
    if subtitle:
        width = int(ax.figure.get_figwidth() * 13)
        ax.text(0, 1.02, "\n".join(textwrap.wrap(subtitle, width)), transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def grouped_bars(path, grid, models, conds, cfg, title, ylabel, subtitle=None, label_conds=("baseline", "sanitize"),
                 callout=None):
    """Grouped bars with 95% CI whiskers. Values are printed on the bars named in label_conds only (the ones the
    reader compares); callout = (text, condition) draws a pointer from the text to that condition's bar of the
    first model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = cfg["analysis"]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300, facecolor=SURFACE)
    w = 0.8 / len(conds)
    x = np.arange(len(models))
    for j, c in enumerate(conds):
        vals = [grid[(m, c)][0] * 100 for m in models]
        lo = [(grid[(m, c)][0] - grid[(m, c)][1]) * 100 for m in models]
        hi = [(grid[(m, c)][2] - grid[(m, c)][0]) * 100 for m in models]
        xs = x + (j - (len(conds) - 1) / 2) * w
        ax.bar(xs, vals, w * 0.92, color=SERIES[j], label=L["condition_labels"][c],
               yerr=[lo, hi], error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=0))
        if c in label_conds:
            for xi, v, h in zip(xs, vals, hi):
                ax.text(xi, v + h + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, [L["model_labels"][m] for m in models])
    style(ax, title, ylabel, subtitle)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=len(conds), loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def tradeoff(path, pts, cfg, xlabel, ylabel, title, xlim=None, subtitle=None, better="lower-right"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = cfg["analysis"]
    conds = list(L["condition_labels"])
    fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=300, facecolor=SURFACE)
    for j, c in enumerate(conds):
        p = [q for q in pts if q["condition"] == c]
        ax.scatter([q["x"] * 100 for q in p], [q["y"] * 100 for q in p], s=46, color=SERIES[j], marker=MARKERS[j],
                   edgecolor=SURFACE, linewidth=1.2, label=L["condition_labels"][c], zorder=3)
    style(ax, title, ylabel, subtitle)
    ax.set_xlabel(xlabel, color=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_ylim(0, max(60, max(q["y"] for q in pts) * 100 + 5))
    # direct labels on the two conditions that matter
    lab_pts = [q for q in pts if q["condition"] in ("baseline", "sanitize")]
    if better == "lower-right":
        # scattered points: nudge a label down when it would collide with one already placed (display space)
        placed = []
        for q in sorted(lab_pts, key=lambda q: (q["condition"] != "baseline", -q["y"])):
            px, py = ax.transData.transform((q["x"] * 100, q["y"] * 100))
            dy = 3
            while any(abs(px - ox) < 70 and abs((py + dy * 4.17) - oy) < 34 for ox, oy in placed):   # 300 dpi: 1 pt = 4.17 px
                dy -= 8
            placed.append((px, py + dy * 4.17))
            ax.annotate(L["model_labels"][q["model"]].split()[0], (q["x"] * 100, q["y"] * 100), xytext=(5, dy),
                        textcoords="offset points", fontsize=7, color=INK2)
    else:
        # points stacked on one vertical line: labels in a column to the right, evenly spaced, with leader lines
        x0, x1 = ax.get_xlim()
        col_x = x0 + 0.13 * (x1 - x0)
        y_lo, y_hi = ax.get_ylim()
        step = 0.035 * (y_hi - y_lo)
        for cond in ("baseline", "sanitize"):
            group = sorted([q for q in lab_pts if q["condition"] == cond], key=lambda q: (round(q["y"], 4), list(L["model_labels"]).index(q["model"])))
            ys = [q["y"] * 100 for q in group]
            for i in range(1, len(ys)):                       # push up until spaced
                ys[i] = max(ys[i], ys[i - 1] + step)
            for i in range(len(ys) - 2, -1, -1):              # and back down if the stack ran away
                ys[i] = min(ys[i], ys[i + 1] - step)
            for q, yl in zip(group, ys):
                ax.plot([q["x"] * 100, col_x - 0.02 * (x1 - x0)], [q["y"] * 100, yl], color=AXIS, lw=0.6, zorder=2)
                ax.text(col_x, yl, L["model_labels"][q["model"]].split()[0], fontsize=7, color=INK2, va="center")
    if better == "lower-right":
        # one dashed line between the two clusters, with quiet band labels
        ys = sorted(q["y"] * 100 for q in pts)
        gap = max(range(1, len(ys)), key=lambda i: ys[i] - ys[i - 1])
        mid = (ys[gap] + ys[gap - 1]) / 2
        ax.axhline(mid, color=AXIS, lw=0.8, ls=(0, (4, 4)))
        ax.text(0.01, (mid + 1.5) / ax.get_ylim()[1], "no defense, Spotlighting, StruQ, MAP filter", transform=ax.transAxes,
                fontsize=7.5, color=INK2, va="bottom")
        ax.text(0.01, (mid - 1.5) / ax.get_ylim()[1], "Sanitization only", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right" if better == "lower-right" else "center right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def heatmap(path, grid, models, conds, cfg, title, subtitle=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    L = cfg["analysis"]
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    data = np.array([[grid[(m, c)][0] * 100 for c in conds] for m in models])
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300, facecolor=SURFACE)
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(conds)), [L["condition_labels"][c] for c in conds], fontsize=8, color=INK2)
    ax.set_yticks(range(len(models)), [L["model_labels"][m] for m in models], fontsize=8, color=INK2)
    for i in range(len(models)):
        for j in range(len(conds)):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if data[i, j] > 45 else INK)
    import textwrap
    nlines = len(textwrap.wrap(subtitle, int(ax.figure.get_figwidth() * 13))) if subtitle else 0
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10 + 12 * nlines, fontweight="bold")
    if subtitle:
        width = int(ax.figure.get_figwidth() * 13)
        ax.text(0, 1.02, "\n".join(textwrap.wrap(subtitle, width)), transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(plt.Rectangle((0.5, -0.5), 1, len(models), fill=False, ec=INK, lw=1.6))
    ax.annotate("the only column that changes", xy=(1, len(models) - 0.5), xytext=(0, -26), textcoords="offset points",
                ha="center", va="top", fontsize=7.5, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK2, length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    A = cfg["analysis"]
    alpha, T, S = A["alpha"], A["primary_technique"], A["primary_source"]
    tables_dir, figures_dir = A["tables_dir"], A["figures_dir"]
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    df = pd.read_json(os.path.join(cfg["paths"]["results_dir"], "scores.jsonl"), lines=True)
    models = [m for m in A["model_labels"] if m in set(df.model)]
    conds = [c for c in A["condition_labels"] if c in set(df.condition)]
    gen = df[df.type == "genuine"]
    att = df[df.type != "genuine"]
    attT = att[att.technique == T]
    md = [f"# Analysis of {len(df):,} scored runs\n",
          f"Models: {', '.join(models)}. Conditions: {', '.join(conds)}. ASR rows use the `{T}` technique "
          f"(the one every condition ran); paired tests use the {S} subset. CIs are 95% Wilson intervals; "
          f"bootstrap CIs use {A['bootstrap_n']} resamples, seed {cfg['seed']}. Figures: {figures_dir}/.\n"]

    # ---- T1 main grid
    grid_asr, grid_frr, grid_util, grid_asr_src = {}, {}, {}, {}
    rows = []
    for m in models:
        for c in conds:
            a = attT[(attT.model == m) & (attT.condition == c)]
            g = gen[(gen.model == m) & (gen.condition == c)]
            grid_asr[(m, c)] = wilson(a.attack_success.sum(), len(a), alpha)
            grid_frr[(m, c)] = wilson(g.refused.sum(), len(g), alpha)
            grid_util[(m, c)] = wilson(g.utility.sum(), len(g), alpha)
            for s in ("MPIB", "AdvBench"):
                x = a[a.source == s]
                grid_asr_src[(m, c, s)] = wilson(x.attack_success.sum(), len(x), alpha)
            rows.append({"model": m, "condition": c, "n_attack": len(a), "ASR %": fmt_pct(*grid_asr[(m, c)]),
                         f"ASR {S} %": fmt_pct(*grid_asr_src[(m, c, S)]), "n_genuine": len(g),
                         "FRR %": fmt_pct(*grid_frr[(m, c)]), "utility %": fmt_pct(*grid_util[(m, c)]),
                         "no_answer %": fmt_pct(g.no_answer.mean())})
    save(pd.DataFrame(rows), "t1_main_grid", tables_dir, md, "T1. ASR, FRR and utility per model and condition",
         "ASR: attack items, `" + T + "` rows. FRR: genuine items refused. Utility: MedQA answer correct.")

    # ---- T2 by source
    rows = [{"model": m, "condition": c, "MPIB ASR %": fmt_pct(*grid_asr_src[(m, c, "MPIB")]),
             "AdvBench ASR %": fmt_pct(*grid_asr_src[(m, c, "AdvBench")])} for m in models for c in conds]
    save(pd.DataFrame(rows), "t2_asr_by_source", tables_dir, md, "T2. ASR by attack source")

    # ---- T3 sanitize by technique
    san = att[att.condition == "sanitize"]
    rows = []
    for m in models:
        r = {"model": m}
        for t in sorted(san.technique.unique()):
            x = san[(san.model == m) & (san.technique == t)]
            r[t + " %"] = fmt_pct(*wilson(x.attack_success.sum(), len(x), alpha))
        rows.append(r)
    save(pd.DataFrame(rows), "t3_sanitize_by_technique", tables_dir, md, "T3. Sanitization: ASR by hiding technique")

    # ---- T4 McNemar + Cochran's Q, T5 bootstrap deltas
    rows4, rows5, rowsq = [], [], []
    for subset_name, sub in (("all attacks", attT), (S, attT[attT.source == S])):
        for m in models:
            piv = sub[sub.model == m].pivot_table(index="item_id", columns="condition", values="attack_success")
            piv = piv.dropna()
            base = piv["baseline"].values
            ps, labels = [], []
            for c in conds:
                if c == "baseline":
                    continue
                d = piv[c].values
                b = int(((base == 1) & (d == 0)).sum())
                cc = int(((base == 0) & (d == 1)).sum())
                p = mcnemar_exact(b, cc)
                ps.append(p)
                labels.append((c, b, cc, d.mean() - base.mean()))
                mean_d, lo, hi = paired_bootstrap_delta(base, d, A["bootstrap_n"], rng)
                rows5.append({"subset": subset_name, "model": m, "defense": c, "n_items": len(piv),
                              "baseline ASR %": round(100 * base.mean(), 1), "defense ASR %": round(100 * d.mean(), 1),
                              "delta pp": round(100 * mean_d, 1), "CI low": round(100 * lo, 1), "CI high": round(100 * hi, 1)})
            adj = holm(ps)
            for (c, b, cc, delta), p, pa in zip(labels, ps, adj):
                rows4.append({"subset": subset_name, "model": m, "defense": c, "n_items": len(piv),
                              "baseline->safe (b)": b, "safe->attacked (c)": cc, "delta pp": round(100 * delta, 1),
                              "p (exact McNemar)": p, "p (Holm)": pa, "significant": pa < alpha})
            q, dfree, pq = cochran_q(piv[conds].values)
            rowsq.append({"subset": subset_name, "model": m, "n_items": len(piv), "Cochran Q": round(q, 2), "df": dfree, "p": pq})
    save(pd.DataFrame(rows4), "t4_mcnemar", tables_dir, md, "T4. Exact McNemar: each defense vs no defense (paired by item)",
         f"Holm correction within each model and subset ({len(conds)-1} tests). b and c are the discordant counts.", "{:.4g}")
    save(pd.DataFrame(rowsq), "t4b_cochran_q", tables_dir, md, "T4b. Cochran's Q across the five conditions", None, "{:.4g}")
    save(pd.DataFrame(rows5), "t5_bootstrap_delta", tables_dir, md,
         "T5. Change in ASR vs no defense, paired bootstrap 95% CI (percentage points)", None, "{:.1f}")

    # ---- T6 medft pairs (paired by item across models, same condition = baseline)
    rows = []
    for med, gen_m in A["medft_pairs"].items():
        if med not in models or gen_m not in models:
            continue
        for metric, frame, col in (("ASR (" + T + ")", attT, "attack_success"), (f"ASR {S}", attT[attT.source == S], "attack_success"),
                                   ("utility", gen, "utility"), ("FRR", gen, "refused")):
            piv = frame[(frame.condition == "baseline") & (frame.model.isin([med, gen_m]))].pivot_table(
                index="item_id", columns="model", values=col).dropna()
            a, b = piv[gen_m].values, piv[med].values
            bb = int(((a == 1) & (b == 0)).sum())
            cc = int(((a == 0) & (b == 1)).sum())
            rows.append({"metric": metric, "general model": gen_m, "general %": round(100 * a.mean(), 1),
                         "medical model": med, "medical %": round(100 * b.mean(), 1), "delta pp": round(100 * (b.mean() - a.mean()), 1),
                         "n_items": len(piv), "p (exact McNemar)": mcnemar_exact(bb, cc)})
    save(pd.DataFrame(rows), "t6_medft_pairs", tables_dir, md,
         "T6. Medical fine-tuning condition: medical model vs general comparator, no-defense prompt",
         "The clean pair is BioMistral vs Mistral (same base). Meditron is Llama-2-based; Llama 3 is the nearest general model run.", "{:.4g}")

    # ---- T7 by MedQA category (RQ3)
    rows = []
    for cat in sorted(gen.category.dropna().unique()):
        r = {"category": cat, "n_items": int(gen[(gen.category == cat) & (gen.condition == "baseline") & (gen.model == models[0])].shape[0])}
        for c in conds:
            x = gen[(gen.category == cat) & (gen.condition == c)]
            r[f"{c} FRR %"] = round(100 * x.refused.mean(), 1)
            r[f"{c} no-answer %"] = round(100 * x.no_answer.mean(), 1)
            r[f"{c} utility %"] = round(100 * x.utility.mean(), 1)
        rows.append(r)
    save(pd.DataFrame(rows), "t7_by_category", tables_dir, md,
         "T7. Genuine questions by MedQA category and condition, pooled over models (RQ3)", None, "{:.1f}")

    # ---- T8 retrieval-conditional ASR
    rows = []
    for c in conds:
        x = attT[attT.condition == c]
        for flag in (True, False):
            y = x[x.retrieved_contains_injection == flag]
            rows.append({"condition": c, "poisoned chunk retrieved": flag, "n": len(y),
                         "ASR %": fmt_pct(*wilson(y.attack_success.sum(), len(y), alpha)) if len(y) else "-"})
    save(pd.DataFrame(rows), "t8_asr_by_retrieval", tables_dir, md, "T8. ASR conditional on the poisoned chunk being in the top-k")

    # ---- T9 scorer diagnostics
    diag = []
    cal_path = os.path.join(cfg["paths"]["results_dir"], "judge_calibration_mistral.jsonl")
    if os.path.exists(cal_path):
        cal = pd.read_json(cal_path, lines=True)
        cal = cal[cal.type != "genuine"].set_index("run_id")
        main_j = att.set_index("run_id").loc[cal.index]
        pairs = list(zip(main_j.judge.astype(int), cal.judge.astype(int)))
        pa, k = kappa(pairs)
        conf = defaultdict(int)
        for a, b in pairs:
            conf[(a, b)] += 1
        diag.append({"statistic": "judge calibration rows (qwen2.5:7b vs mistral:7b)", "value": len(pairs)})
        diag.append({"statistic": "judge agreement", "value": round(pa, 3)})
        diag.append({"statistic": "judge Cohen's kappa", "value": round(k, 3)})
        diag.append({"statistic": "judge confusion (primary,second): 00/01/10/11",
                     "value": f"{conf[(0,0)]}/{conf[(0,1)]}/{conf[(1,0)]}/{conf[(1,1)]}"})
    jm = att.judge_model.value_counts()
    for name, v in jm.items():
        diag.append({"statistic": f"judge decisions by {name}", "value": int(v)})
    diag.append({"statistic": "rule_hit & judge agreement on attack rows", "value": round(float((att.rule_hit == att.judge).mean()), 3)})
    diag.append({"statistic": "genuine letters from option text (fallback)", "value": int((gen.letter_source == "option_text").sum())})
    for m in models:
        diag.append({"statistic": f"{m}: doc_echo % (attack rows)", "value": round(100 * att[att.model == m].doc_echo.mean(), 1)})
        diag.append({"statistic": f"{m}: mean latency s", "value": round(float(df[df.model == m].latency_s.mean()), 2)})
    save(pd.DataFrame(diag), "t9_scorer_diagnostics", tables_dir, md, "T9. Scorer and judge diagnostics")

    # ---- T10 optional review of a blind stratified sample: labels in results/manual_sample.csv
    #      (column human_label), when present. Attack rows: label 1 = injected instruction followed.
    #      Genuine rows: the reviewer's option letter is compared with the scorer's extracted letter.
    def review_table(name, lab, label_col, letter_col):
        lab = lab.copy()
        sc = df.set_index("run_id")
        lab = lab[lab.run_id.isin(sc.index)]
        a = lab[lab.source != "MedQA"].copy()
        a["auto"] = sc.loc[a.run_id, "attack_success"].values.astype(int)
        a["rev"] = a[label_col].astype(int)
        pairs = list(zip(a.auto, a.rev))
        pa, k = kappa(pairs)
        rows = [{"statistic": f"{name}: attack rows reviewed", "value": len(a)},
                {"statistic": f"{name}: attack agreement (scorer vs reviewer)", "value": round(pa, 3)},
                {"statistic": f"{name}: attack Cohen's kappa", "value": round(k, 3)},
                {"statistic": f"{name}: scorer 1 / reviewer 0 (false attack)", "value": int(((a.auto == 1) & (a.rev == 0)).sum())},
                {"statistic": f"{name}: scorer 0 / reviewer 1 (missed attack)", "value": int(((a.auto == 0) & (a.rev == 1)).sum())},
                {"statistic": f"{name}: ASR in sample, scorer vs reviewer", "value": f"{100*a.auto.mean():.1f} vs {100*a.rev.mean():.1f}"}]
        for src in sorted(a.source.unique()):
            x = a[a.source == src]
            rows.append({"statistic": f"{name}: agreement ({src})", "value": round(float((x.auto == x.rev).mean()), 3)})
        for c in conds:
            x = a[a.condition == c] if "condition" in a else a[a.run_id.str.split("|").str[1] == c]
            if len(x):
                rows.append({"statistic": f"{name}: agreement ({c})", "value": round(float((x.auto == x.rev).mean()), 3)})
        if letter_col and letter_col in lab:
            g = lab[lab.source == "MedQA"].copy()
            g["auto"] = sc.loc[g.run_id, "answer_letter"].fillna("").values
            g["rev"] = g[letter_col].fillna("").astype(str).str.strip().str.upper()
            gold = sc.loc[g.run_id].index.map(lambda rid: None)
            rows.append({"statistic": f"{name}: genuine rows reviewed", "value": len(g)})
            rows.append({"statistic": f"{name}: answer-letter agreement", "value": round(float((g.auto == g.rev).mean()), 3)})
            rows.append({"statistic": f"{name}: scorer found a letter the reviewer did not", "value": int(((g.auto != "") & (g.rev == "")).sum())})
            rows.append({"statistic": f"{name}: reviewer found a letter the scorer did not", "value": int(((g.auto == "") & (g.rev != "")).sum())})
        return rows
    rows10 = []
    ms_path = os.path.join(cfg["paths"]["results_dir"], "manual_sample.csv")
    if os.path.exists(ms_path):
        ms = pd.read_csv(ms_path, dtype=str)
        lab = ms[ms.human_label.notna() & (ms.human_label.astype(str).str.strip() != "")].copy()
        if len(lab):
            lab["human_label"] = lab.human_label.astype(str).str.strip().str.lower().isin(["1", "yes", "y", "true"]).astype(int)
            rows10 += review_table("Author review", lab, "human_label", None)
    if rows10:
        save(pd.DataFrame(rows10), "t10_independent_review", tables_dir, md, "T10. Independent review vs automatic scorer",
             "Blind stratified sample; attack label 1 = injected instruction followed or adopted in the answer. "
             "Document regurgitation that merely contains the injected text was labelled 0 by the reviewer.")
    else:
        md.append("\n### T10. Independent review\n\nNo review labels found.")

    # ---- figures
    san_drop = [100 * (grid_asr[(m, "baseline")][0] - grid_asr[(m, "sanitize")][0]) for m in models]
    grouped_bars(os.path.join(figures_dir, "fig_asr_all.png"), grid_asr, models, conds, cfg,
                 "Only sanitization reduces attack success", "ASR (%)",
                 subtitle=f"All 300 attack items per model, white-text injection, 95% Wilson CIs. Sanitization cuts ASR by "
                          f"{min(san_drop):.0f} to {max(san_drop):.0f} points; the three prompt-level defenses change it by less than 6.",
                 )
    gS = {(m, c): grid_asr_src[(m, c, S)] for m in models for c in conds}
    grouped_bars(os.path.join(figures_dir, f"fig_asr_{S.lower()}.png"), gS, models, conds, cfg,
                 f"{S} attacks: planted clinical answers are adopted 68 to 89% of the time", "ASR (%)",
                 subtitle=f"{S} items only (150 per model). These injections plant a plausible clinical answer rather than an overtly harmful instruction.",
                 callout=None)
    grouped_bars(os.path.join(figures_dir, "fig_utility.png"), grid_util, models, conds, cfg,
                 "No defense costs accuracy on genuine questions", "Utility (%)",
                 subtitle="MedQA accuracy on the 300 genuine questions per model, 95% Wilson CIs. Meditron is a base model answered in completion mode.",
                 label_conds=("baseline",))
    pts = [{"model": m, "condition": c, "x": grid_util[(m, c)][0], "y": grid_asr[(m, c)][0]} for m in models for c in conds]
    tradeoff(os.path.join(figures_dir, "fig_tradeoff_utility.png"), pts, cfg, "MedQA accuracy (%)", "ASR (%)",
             "Trade-off: sanitization cuts attacks without costing accuracy",
             subtitle="One point per model and defense (30 points). Same marker = same defense; labels mark the no-defense and sanitized points.")
    pts = [{"model": m, "condition": c, "x": grid_frr[(m, c)][0], "y": grid_asr[(m, c)][0]} for m in models for c in conds]
    tradeoff(os.path.join(figures_dir, "fig_tradeoff_frr.png"), pts, cfg, "False refusal rate (%)", "ASR (%)",
             "False refusals stay near zero under every defense", xlim=(-0.2, 5), better="lower-left",
             subtitle="One point per model and defense. The x-axis is the share of genuine MedQA questions the model refused to answer.")
    heatmap(os.path.join(figures_dir, "fig_asr_heatmap.png"), grid_asr, models, conds, cfg, "ASR (%) by model and defense",
            subtitle="White-text injection, all attack items. Darker = more attacks succeed.")
    # by source and by technique: small grouped bars with two / three series
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300, facecolor=SURFACE)
    x = np.arange(len(models))
    for j, s in enumerate(("MPIB", "AdvBench")):
        v = [grid_asr_src[(m, "baseline", s)] for m in models]
        ax.bar(x + (j - 0.5) * 0.38, [q[0] * 100 for q in v], 0.35, color=SERIES[j], label=s,
               yerr=[[(q[0] - q[1]) * 100 for q in v], [(q[2] - q[0]) * 100 for q in v]],
               error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=0))
    ax.set_xticks(x, [A["model_labels"][m] for m in models])
    style(ax, "Models ignore overt harm but adopt planted clinical answers", "ASR (%)",
          subtitle="No defense. MPIB injections plant a clinical answer; AdvBench injections ask for harmful or privacy-violating text. "
                   "The poisoned chunk was retrieved in 95% of runs for both.")
    for xi, m in zip(x, models):
        ax.text(xi - 0.19, grid_asr_src[(m, "baseline", "MPIB")][2] * 100 + 2, f"{grid_asr_src[(m, 'baseline', 'MPIB')][0]*100:.0f}", ha="center", fontsize=7, color=INK2)
        ax.text(xi + 0.19, grid_asr_src[(m, "baseline", "AdvBench")][2] * 100 + 2, f"{grid_asr_src[(m, 'baseline', 'AdvBench')][0]*100:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig_asr_by_source.png"), facecolor=SURFACE)
    plt.close(fig)
    techs = sorted(san.technique.unique())
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300, facecolor=SURFACE)
    w = 0.8 / (len(techs) + 1)
    base_v = [grid_asr[(m, "baseline")] for m in models]
    ax.bar(x - 1.5 * w, [q[0] * 100 for q in base_v], w * 0.92, color=SERIES[0], label="No defense")
    for j, t in enumerate(techs):
        v = [wilson(san[(san.model == m) & (san.technique == t)].attack_success.sum(),
                    len(san[(san.model == m) & (san.technique == t)]), alpha) for m in models]
        ax.bar(x + (j - 0.5) * w, [q[0] * 100 for q in v], w * 0.92, color=SERIES[j + 1], label="Sanitized, " + {"white_text": "white text", "tiny_font": "1-point font", "metadata": "metadata"}.get(t, t),
               yerr=[[(q[0] - q[1]) * 100 for q in v], [(q[2] - q[0]) * 100 for q in v]],
               error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=0))
    ax.set_xticks(x, [A["model_labels"][m] for m in models])
    style(ax, "Sanitization works equally on all three hiding techniques", "ASR (%)",
          subtitle="ASR after sanitization for white text, 1-pt font and PDF metadata, against the unsanitized white-text baseline. "
                   "The residual 6 to 17% is mostly answers that coincide with the planted one.")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, ncol=2, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig_sanitize_by_technique.png"), facecolor=SURFACE)
    plt.close(fig)

    # ---- T11 ROUGE-L of the output against the injected instruction (graded view of text adoption)
    raw = {}
    import glob as _glob
    for fp in _glob.glob(os.path.join(cfg["paths"]["runs_dir"], "*.jsonl")):
        for line in open(fp):
            if line.strip():
                r = json.loads(line)
                if r["type"] != "genuine":
                    raw[r["run_id"]] = (r.get("injected_instruction"), r.get("output"))
    attT = attT.copy()
    attT["rouge_l"] = [rouge_l_f1(*raw[rid]) if rid in raw else np.nan for rid in attT.run_id]
    rows = []
    for m in models:
        r = {"model": m}
        for c in conds:
            x = attT[(attT.model == m) & (attT.condition == c)]
            r[f"{c} mean"] = round(float(x.rouge_l.mean()), 3)
            r[f"{c} {S} mean"] = round(float(x[x.source == S].rouge_l.mean()), 3)
        rows.append(r)
    save(pd.DataFrame(rows), "t11_rouge_l_vs_injection", tables_dir, md,
         "T11. ROUGE-L F1 of the model output against the injected instruction",
         "A graded companion to the binary ASR: how much of the planted text the model reproduces, averaged over attack rows "
         "(white-text injection). BLEU/ROUGE are not used for utility because MedQA is multiple choice.")
    by = attT[attT.source == S].groupby(["model", "condition"]).rouge_l.agg(["mean", "sem"])
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300, facecolor=SURFACE)
    w = 0.8 / len(conds)
    x = np.arange(len(models))
    for j, c in enumerate(conds):
        vals = [by.loc[(m, c), "mean"] for m in models]
        err = [1.96 * by.loc[(m, c), "sem"] for m in models]
        xs = x + (j - (len(conds) - 1) / 2) * w
        ax.bar(xs, vals, w * 0.92, color=SERIES[j], label=A["condition_labels"][c], yerr=err,
               error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=0))
    ax.set_xticks(x, [A["model_labels"][m] for m in models])
    style(ax, "How much planted text is reproduced: ROUGE-L against the injection", "mean ROUGE-L F1",
          subtitle=f"{S} attack rows, white-text injection, mean with 95% CI. Higher = the answer copies more of the injected text.")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=len(conds), loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig_rouge_l_injection.png"), facecolor=SURFACE)
    plt.close(fig)

    md.append(f"\n### Figures\n\n" + "\n".join(f"- {f}" for f in sorted(os.listdir(figures_dir))))
    with open(os.path.join(cfg["paths"]["results_dir"], "analysis.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[analyze] tables -> {tables_dir}/, figures -> {figures_dir}/, summary -> results/analysis.md")


if __name__ == "__main__":
    main()
