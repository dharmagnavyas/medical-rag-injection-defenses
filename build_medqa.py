import json, random, re, os
from collections import Counter
from datasets import load_dataset

SEED = 42
N = 300
os.makedirs("data/benchmark", exist_ok=True)

ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
random.seed(SEED)
idx = random.sample(range(len(ds)), N)

# First matching rule wins. Applied to the question stem only.
RULES = [
    ("management",      r"next (best )?step|most appropriate (next step|management|treatment|therapy|intervention|pharmacotherapy|initial|procedure|response)|initial (treatment|management|therapy|step)|best (initial )?(treatment|therapy|management|long-term)|treatment of choice|treatment option|optimal intervention|prophylaxis|vaccine|should (also )?be (administered|treated|managed|considered)|appropriate treatment|could have prevented"),
    ("pharmacology",    r"(drug|medication|agent|pharmacotherap|dose|dosage|mechanism of action|adverse effect|side effect|contraindicat|prescribed|disinfectant)"),
    ("investigation",   r"most appropriate (test|investigation|diagnostic|assessment)|which (test|study|imaging)|most sensitive test|confirm the diagnosis|best (test|study)|screening|additional history|tests should|further assessment"),
    ("risk_prognosis",  r"at (greatest |increased |most )?risk|risk factor|predispos|prognosis|complication|most at risk|without treatment"),
    ("biostat_ethics",  r"type (1|2|i|ii) error|study|experiment|research|statistic|mode of|median|mean of|cut-off|sensitivity|specificity|inference|ACGME|compensation|editor|association between|primary measure|physician"),
    ("diagnosis",       r"most likely diagnosis|most likely (cause|etiology|malignant|tumor|pathology)|diagnos|most consistent with|accurately describes|accurate description"),
    ("findings",        r"most likely (to show|finding|to be found|to reveal|seen|to demonstrate|be found|be seen|occur)|(finding|findings) (will|would|do|is|are|in)|expected to (show|find|see)|would most likely be (seen|found)|will most likely be found|most likely be anticipated|expect to see|be observed|likely elevated|ECG findings|imaging finding|cerebrospinal fluid"),
    ("pathophysiology", r"mechanism|pathophysiolog|underlying|histolog|explains|explanation|responsible for|accounts for|which of the following (processes|changes)|due to a mutation|most likely cause|is causing|caused this|disrupted|lesions? would result|contributed to"),
    ("basic_science",   r"receptor|enzyme|substance|molecule|cells?\b|gene|protein|hormone|virulence|structure|nerve|artery|vein|vessel|muscle|organism|pathogen|virus|antibod|immunoglobulin|cytokine|channel|transporter|mediator|neurotransmitter|stimulates|inhibits|secreted|released|oxygen content|gas exchange|loci|linked|skin areas|spreads to"),
    ("other",           r"."),
]

def stem(text):
    """Return the question stem: the last line ending in ? or :, else the last sentence."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for l in reversed(lines):
        if l.endswith("?") or l.endswith('?"') or l.endswith(":"):
            return l
    return re.split(r"(?<=[.?!])\s+", text.strip())[-1]

def tag(text):
    s = stem(text).lower()
    for name, pat in RULES[:-1]:
        if re.search(pat, s):
            return name
    return "other"

items = []
for n, i in enumerate(idx):
    q = ds[i]
    items.append({
        "id": f"MEDQA_{n:03d}",
        "source": "MedQA",
        "type": "genuine",
        "category": tag(q["question"]),
        "question": q["question"],
        "options": q["options"],
        "answer_idx": q["answer_idx"],
        "answer": q["answer"],
    })

with open("data/benchmark/medqa_300.jsonl", "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")

print(Counter(it["category"] for it in items))

with open("data/benchmark/other_stems.txt", "w") as f:
    for it in items:
        if it["category"] == "other":
            f.write(stem(it["question"]) + "\n")