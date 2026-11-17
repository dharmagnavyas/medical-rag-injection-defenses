"""Step 2d: one synthetic discharge summary PDF per benchmark item, plus attacked variants.

Clean:    data/docs/clean/<id>.pdf          (all items)
Attacked: data/docs/<technique>/<id>.pdf    (attack items only; techniques from config.yaml:
          white_text = body-size white text, tiny_font = 1-pt black text, metadata = PDF Subject field)
Manifest: data/docs/manifest.jsonl, one row per PDF. Resume-safe: existing PDFs are not rewritten.
After writing, every PDF is read back with PyMuPDF to confirm the injected text is present (attacked)
or absent (clean). No real patient data: bodies are built from the fictional exam vignettes.
"""
import hashlib, json, os, random, re, sys
import pymupdf
from reportlab.lib.colors import black, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from bench import load_config, load_items

cfg = load_config()
D = cfg["documents"]
DOCS = cfg["paths"]["docs_dir"]
TECHS = D["hiding_techniques"]
BODY_PT = D["body_font_pt"]

st_body = ParagraphStyle("body", fontName="Helvetica", fontSize=BODY_PT, leading=BODY_PT * 1.3)
st_head = ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=BODY_PT + 2, leading=(BODY_PT + 2) * 1.3)
st_sec = ParagraphStyle("sec", fontName="Helvetica-Bold", fontSize=BODY_PT, leading=BODY_PT * 1.3, spaceBefore=4)
st_white = ParagraphStyle("white", parent=st_body, textColor=white)
st_tiny = ParagraphStyle("tiny", fontName="Helvetica", fontSize=D["tiny_font_pt"], leading=D["tiny_font_pt"] * 1.2, textColor=black)

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def split_vignette(question):
    """Return (history text, question stem).

    The stem is the final question sentence ("Which of the following ...?"). Most MedQA items are a single
    paragraph, so we split on sentences, not lines: everything before the last sentence ending in '?' is
    history. Lab-value lines (no terminal punctuation) are kept inside the history block.
    """
    text = question.strip()
    # sentence boundaries: ., ?, ! followed by whitespace, or a newline
    parts = [p.strip() for p in re.split(r"(?<=[.?!])\s+|\n+", text) if p.strip()]
    stem_idx = None
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].endswith(("?", '?"')):
            stem_idx = i
            break
    if stem_idx is None or stem_idx == 0:
        return (text if stem_idx is None else ""), (parts[0] if parts and stem_idx == 0 else "")
    # the stem may run over several short fragments after a lab table; take from the last '?' sentence only
    stem = parts[stem_idx]
    history = " ".join(parts[:stem_idx] + parts[stem_idx + 1:])
    return history, stem

def summary_sections(item, rng):
    """Synthetic discharge summary built from the fictional vignette."""
    history, _ = split_vignette(item["question"])
    mrn = f"SYN-{rng.randint(100000, 999999)}"
    day = rng.randint(1, 28); month = rng.randint(1, 12); los = rng.randint(1, 6)
    adm = f"{day:02d}/{month:02d}/2025"; dis = f"{min(day + los, 28):02d}/{month:02d}/2025"
    return [
        ("Discharge Summary", None),
        ("Patient identifier", f"{mrn} (synthetic record, generated for research; no real patient)"),
        ("Admission date", adm), ("Discharge date", dis),
        ("Presenting history and findings", history),
        ("Hospital course", "Assessed by the admitting team. Investigations and management were carried out "
                            "according to the department protocol for the presenting problem. The patient's "
                            "condition was reviewed daily."),
        ("Medications on discharge", "As per the discharge prescription issued by the treating team."),
        ("Follow-up", f"Outpatient review in {rng.choice([1, 2, 4, 6])} weeks. Return earlier if symptoms worsen."),
    ]

def build_pdf(path, sections, injection=None, technique=None):
    subject = "Discharge summary"
    if technique == "metadata":
        subject = injection
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Discharge Summary", author="Synthetic data generator", subject=subject)
    flow = []
    for head, body in sections:
        if body is None:
            flow += [Paragraph(esc(head), st_head), Spacer(1, 6)]
        else:
            flow += [Paragraph(esc(head), st_sec), Paragraph(esc(body), st_body)]
        # hidden text sits right after the history block, where a reader would least expect it
        if head == "Presenting history and findings" and technique in ("white_text", "tiny_font"):
            flow.append(Paragraph(esc(injection), st_white if technique == "white_text" else st_tiny))
    doc.build(flow)

def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()

def verify(path, injection, expect_present):
    # very short injections (e.g. "10cc") can occur naturally in the vignette; only the attacked check is meaningful
    if not expect_present and len(norm(injection)) < 25:
        return
    with pymupdf.open(path) as d:
        text = " ".join(p.get_text() for p in d) + " " + " ".join(str(v) for v in d.metadata.values() if v)
    present = norm(injection) in norm(text)
    if present != expect_present:
        raise SystemExit(f"verify failed: {path} injected text present={present}, expected {expect_present}")

def main():
    items = load_items(cfg)
    for sub in ["clean"] + TECHS:
        os.makedirs(os.path.join(DOCS, sub), exist_ok=True)
    manifest, written = [], 0
    for it in items:
        rng = random.Random(f"{cfg['seed']}:{it['id']}")
        sections = summary_sections(it, rng)
        inj = it.get("injected_instruction")
        variants = [("clean", None)] + ([(t, inj) for t in TECHS] if it["type"] == "attack" else [])
        for tech, text in variants:
            path = os.path.join(DOCS, tech, f"{it['id']}.pdf")
            if not os.path.exists(path):
                build_pdf(path, sections, text, None if tech == "clean" else tech)
                written += 1
            if inj:
                verify(path, inj, expect_present=(tech != "clean"))
            manifest.append({"item_id": it["id"], "source": it["source"], "type": it["type"],
                             "variant": tech, "path": path,
                             "injection_sha256": hashlib.sha256(inj.encode()).hexdigest() if text else None,
                             "n_chars_body": sum(len(b or "") for _, b in sections)})
    with open(cfg["paths"]["docs_manifest"], "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    print(f"{written} PDFs written, {len(manifest)} in manifest -> {cfg['paths']['docs_manifest']}")

if __name__ == "__main__":
    main()
