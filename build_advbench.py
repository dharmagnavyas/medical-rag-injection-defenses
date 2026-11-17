"""Step 2c, part 1: sample 150 AdvBench goals = 6 harm categories x 25 (seed from config.yaml).

Writes data/benchmark/advbench_150_TO_FILL.csv. The medical adaptation (columns medical_injection,
target_phrase) is authored separately and saved as advbench_150.csv with review_status per row.
"""
import re, random, yaml
import pandas as pd

cfg = yaml.safe_load(open("config.yaml"))
SEED, PER_CAT = cfg["seed"], 25
RAW = "data/raw/advbench_harmful_behaviors.csv"
OUT = "data/benchmark/advbench_150_TO_FILL.csv"

# First matching rule wins. Six categories map to six medical harm types (Chapter 3 of the thesis).
CATS = [
    ("misinformation", r"fake news|false information|misinformation|conspiracy|hoax|propaganda|mislead|fake review|fake (article|story|website)|disinformation|deceiv|false (claim|statement)|manipulat.*(opinion|public|election)|vaccine|climate|denial"),
    ("data_theft",     r"steal|hack|exploit|unauthori[sz]ed access|access.*(database|account|network|system)|personal (information|data)|sensitive (information|data)|identity theft|password|credential|keylog|spyware|malware|ransomware|virus|intercept|surveil|track(ing)? (someone|people|a person)|breach|phish|leak|harvest|confidential"),
    ("fraud",          r"fraud|scam|counterfeit|forg|impersonat|insider trading|launder|cheat|tax|embezzl|ponzi|fake (id|identity|document|review|degree|diploma)|plagiari|evade|evasion|bribe|extort|blackmail|manipulate (stock|financial|market)|financial (gain|damage)"),
    ("harassment",     r"harass|bully|threat|stalk|abuse|discriminat|hate|racis|inflammatory|intimidat|humiliat|defam|revenge porn|sexual|troll|slander|exclud|offensive|prejudic|xenophob|sexist"),
    ("substances",     r"drug|alcohol|poison|chemical|toxic|overdose|substance|narcotic|cocaine|meth|opioid|heroin|smok|dosage|medicat|prescription"),
    ("dangerous_acts", r"bomb|weapon|explosive|kill|murder|harm|injur|suicide|self-harm|dangerous|violence|violent|attack|sabotage|arson|fire|crash|terror|shoot|gun|firearm|assassinat|anorexia|bulimia|eating disorder|destroy|damage|disrupt|vandal|riot|hijack|derail|assault|stunt|law enforcement"),
]
def categorise(goal):
    g = goal.lower()
    for name, pat in CATS:
        if re.search(pat, g):
            return name
    return "other"

df = pd.read_csv(RAW)
df["harm_category"] = df["goal"].map(categorise)
print(df["harm_category"].value_counts().to_string())

random.seed(SEED)
rows = []
for name, _ in CATS:
    pool = df.index[df["harm_category"] == name].tolist()
    rows += sorted(random.sample(pool, PER_CAT))
out = df.loc[rows, ["goal", "harm_category"]].reset_index(drop=True)
out.insert(0, "id", [f"ADV_{i:03d}" for i in range(len(out))])
out["medical_injection"] = ""
out["target_phrase"] = ""
out.to_csv(OUT, index=False)
print(f"\nwrote {OUT}: {len(out)} rows, {PER_CAT} per category")
