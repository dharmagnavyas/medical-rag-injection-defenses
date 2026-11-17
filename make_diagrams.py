"""Draw the methodology diagrams for Chapter 3 (not results plots). Reads settings from config.yaml.

Usage: python make_diagrams.py            -> results/figures/fig_3_1_pipeline.png, fig_3_2_hidden_text.png,
                                          fig_3_3_defense_positions.png
"""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import FancyBboxPatch

CFG = yaml.safe_load(open("config.yaml"))
OUT = Path(CFG["analysis"]["figures_dir"])
INK, INK2, SURFACE, RULE = "#1f1f1e", "#5c5c59", "#fcfcfb", "#d9d8d4"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def pipeline():
    r, g, d = CFG["retrieval"], CFG["generation"], CFG["documents"]
    n_models = len(CFG["models"])
    run_conds = [c for c in CFG["conditions"] if c != "medft"]
    stages = [
        ("Benchmark construction", BLUE,
         "300 MedQA genuine questions + 150 MPIB attacks + 150 AdvBench-adapted attacks = 600 items"),
        ("Document generation", BLUE,
         f"One synthetic discharge summary (PDF) per item. Each attack item also in {len(d['hiding_techniques'])} "
         f"poisoned variants: white text, {d['tiny_font_pt']}-pt text, PDF metadata. 1,500 PDFs"),
        ("Retrieval", BLUE,
         f"PyMuPDF text extraction, {r['chunk_chars']}-character chunks, {r['embedder']} embeddings, FAISS, "
         f"top-{r['top_k']} chunks. Precomputed once and reused for every model"),
        ("Defense layer", ORANGE,
         "No defense / visual sanitization / Spotlighting / StruQ-style prompting / MAP-style four-stage filter"),
        ("Generation", BLUE,
         f"{n_models} open-source models of 7 to 8 billion parameters on Ollama, temperature {g['temperature']}, "
         f"seed {g['seed']}. Medical fine-tuning = medical model compared with general model. 21,600 runs"),
        ("Scoring", GREEN,
         "Utility: answer letter correct. FRR: genuine question refused. "
         "ASR: injected instruction followed (rule or judge model)"),
        ("Analysis", GREEN,
         "Wilson confidence intervals, McNemar and Cochran's Q tests, paired bootstrap, "
         "security and utility trade-off"),
    ]
    n = len(stages)
    fig, ax = plt.subplots(figsize=(6.3, 6.6), dpi=300)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    h, gap = 10.6, 3.4
    top = 99
    bx, bw = 1, 33                     # stage box
    tx = bx + bw + 3                   # detail text
    for i, (name, col, detail) in enumerate(stages):
        y = top - (i + 1) * h - i * gap
        ax.add_patch(FancyBboxPatch((bx, y), bw, h, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=col, ec="none"))
        ax.text(bx + 3.2, y + h / 2, str(i + 1), color="white", fontsize=13, fontweight="bold",
                va="center", ha="center")
        ax.text(bx + 6.6, y + h / 2, "\n".join(textwrap.wrap(name, 16)), color="white", fontsize=9.2,
                fontweight="bold", va="center", ha="left", linespacing=1.15)
        ax.text(tx, y + h / 2, "\n".join(textwrap.wrap(detail, 58)), color=INK, fontsize=7.6,
                va="center", ha="left", linespacing=1.3)
        ax.plot([tx - 1.5, 99], [y - gap / 2, y - gap / 2], color=RULE, lw=0.6) if i < n - 1 else None
        if i < n - 1:                  # connector between stage boxes
            cx = bx + bw / 2
            ax.annotate("", xy=(cx, y - gap + 0.3), xytext=(cx, y - 0.3),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0, mutation_scale=9))
    # legend
    ly = 1.2
    for j, (lab, col) in enumerate([("Data and system", BLUE), ("Defense under test", ORANGE),
                                    ("Measurement", GREEN)]):
        x0 = 1 + j * 30
        ax.add_patch(FancyBboxPatch((x0, ly), 3, 2.2, boxstyle="round,pad=0,rounding_size=0.5", fc=col, ec="none"))
        ax.text(x0 + 4.2, ly + 1.1, lab, fontsize=7.4, color=INK2, va="center")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_3_1_pipeline.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / "fig_3_1_pipeline.png")


def hidden_text():
    """Figure 3.2: the same attacked page as a reader sees it and with the white-text span revealed."""
    import io
    import numpy as np
    import pymupdf
    from PIL import Image
    item = CFG["analysis"]["diagram_example_item"]
    path = Path(CFG["paths"]["docs_dir"]) / "white_text" / f"{item}.pdf"
    thr = CFG["sanitize"]["white_threshold"]

    def render(reveal):
        doc = pymupdf.open(path); page = doc[0]
        ymax, boxes = 0, []
        for b in page.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                for sp in ln["spans"]:
                    ymax = max(ymax, sp["bbox"][3])
                    c = sp["color"]; rgb = ((c >> 16) & 255, (c >> 8) & 255, c & 255)
                    if all(1 - v / 255 <= thr for v in rgb):
                        boxes.append(pymupdf.Rect(sp["bbox"]))
        if reveal:
            for r in boxes:
                page.draw_rect(r + (-1.5, -1, 1.5, 1), color=None, fill=(0.92, 0.41, 0.20), overlay=False)
        clip = pymupdf.Rect(page.rect.x0 + 40, page.rect.y0 + 40, page.rect.x1 - 40, ymax + 18)
        pix = page.get_pixmap(dpi=220, clip=clip)
        return np.asarray(Image.open(io.BytesIO(pix.tobytes("png")))), len(boxes)

    a, _ = render(False); b, nb = render(True)
    assert nb > 0, "no white span found"
    asp = a.shape[0] / a.shape[1]
    w = 6.3; pw = (w - 0.35) / 2
    fig = plt.figure(figsize=(w, pw * asp + 1.25), dpi=300); fig.patch.set_facecolor(SURFACE)
    H = fig.get_figheight(); ph = pw * asp / H
    for k, (img, title) in enumerate([(a, "(a) Page as a human reader sees it"),
                                      (b, "(b) Same page with the hidden text revealed")]):
        ax = fig.add_axes([(0.05 + k * (pw + 0.25)) / w, 1 - ph - 0.3 / H, pw / w, ph])
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(RULE); sp.set_linewidth(0.8)
        ax.set_title(title, fontsize=8.2, color=INK, loc="left", pad=5, fontweight="bold")
    d = CFG["documents"]
    key = [("White text", "body-size text in white, placed after the history section (shown in panel b)"),
           (f"{d['tiny_font_pt']}-point text", "the same sentence in the same position, in black at a font size of 1 point"),
           ("PDF metadata", f"the same sentence stored in the {d['metadata_field'].capitalize()} field of the file, not on the page")]
    for j, (k1, k2) in enumerate(key):
        y = (0.72 - j * 0.25) / H
        fig.text(0.05 / w, y, k1, fontsize=7.6, color=INK, fontweight="bold", va="center")
        fig.text(1.15 / w, y, k2, fontsize=7.6, color=INK2, va="center")
    fig.savefig(OUT / "fig_3_2_hidden_text.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / "fig_3_2_hidden_text.png")


def defense_positions():
    """Figure 3.3: one pipeline strip per defense with the stage where it acts highlighted."""
    sz = CFG["sanitize"]
    stages = ["PDF document", "Text extraction", "Retrieval", "Prompt assembly", "Language model"]
    rows = [
        ("Visual sanitization", 1,
         f"Spans below {sz['min_font_pt']} pt or near-white are dropped and metadata is ignored, before indexing"),
        ("Spotlighting", 3,
         "Every space in the retrieved text is replaced by '^'; the system prompt forbids obeying marked text"),
        ("StruQ-style prompting", 3,
         "Retrieved text is placed in [DATA] and the question in [INST]; delimiter-like strings are stripped"),
        ("MAP-style filter", 3,
         "Four-stage filter removes sentences that look like instructions from the retrieved text"),
        ("Medical fine-tuning", 4,
         "No change to the pipeline; a medically fine-tuned model replaces the general model"),
    ]
    GREY, GREYT = "#ecebe7", "#5c5c59"
    fig, ax = plt.subplots(figsize=(6.3, 5.3), dpi=300)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    n = len(rows); rh = 100 / n
    bw, bg, bh = 17.6, 2.6, 6.4
    x0 = 100 - (len(stages) * bw + (len(stages) - 1) * bg) - 0.5
    for i, (name, active, note) in enumerate(rows):
        ytop = 100 - i * rh
        ax.text(0.5, ytop - 3.2, name, fontsize=9, fontweight="bold", color=INK, va="center")
        yb = ytop - 6.2 - bh - 1.2
        for j, st in enumerate(stages):
            x = x0 + j * (bw + bg)
            on = j == active
            ax.add_patch(FancyBboxPatch((x, yb), bw, bh, boxstyle="round,pad=0,rounding_size=1.2",
                                        fc=ORANGE if on else GREY, ec="none"))
            ax.text(x + bw / 2, yb + bh / 2, st, fontsize=7.4, ha="center", va="center",
                    color="white" if on else GREYT, fontweight="bold" if on else "normal")
            if j < len(stages) - 1:
                ax.plot([x + bw + 0.4, x + bw + bg - 0.4], [yb + bh / 2] * 2, color="#b9b8b3", lw=1.0)
        ax.text(x0, yb - 2.6, note, fontsize=7.2, color=INK2, va="center")
        if i < n - 1:
            ax.plot([0.5, 99.5], [ytop - rh + 0.6] * 2, color=RULE, lw=0.6)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT / "fig_3_3_defense_positions.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / "fig_3_3_defense_positions.png")


if __name__ == "__main__":
    pipeline()
    hidden_text()
    defense_positions()
