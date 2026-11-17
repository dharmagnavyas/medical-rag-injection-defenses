import json
from huggingface_hub import hf_hub_download

path = hf_hub_download("jhlee0619/mpib", "data/test.jsonl", repo_type="dataset")
rows = [json.loads(l) for l in open(path)]
print("Rows in test split:", len(rows))
print("Top-level keys:", list(rows[0].keys()))

def show(v, depth=0):
    pad = "  " * depth
    if isinstance(v, dict):
        for k, x in v.items():
            if isinstance(x, (dict, list)):
                print(f"{pad}{k}:"); show(x, depth + 1)
            else:
                print(f"{pad}{k}: {str(x)[:200]}")
    elif isinstance(v, list):
        print(f"{pad}[list of {len(v)}]")
        if v: show(v[0], depth + 1)

from collections import Counter
print("\nvectors:", Counter(r["vector"] for r in rows))
print("scenarios:", Counter(r["scenario"] for r in rows))
print("severity:", Counter(r["labels"]["severity"] for r in rows))
print("harm_types:", Counter(h for r in rows for h in r["labels"]["harm_types"]))

seen = set()
for r in rows:
    v = r["vector"]
    if v == "V0" or v in seen:
        continue
    seen.add(v)
    print(f"\n=== vector {v} | scenario {r['scenario']} | severity {r['labels']['severity']} ===")
    print("user_query:", r["user_query"][:200])
    print("rule:", r["rule"])
    print("harm_types:", r["labels"]["harm_types"])
    print("expected_safe_behavior:", r["labels"]["expected_safe_behavior"])
    for c in r["contexts"][:2]:
        print("  context role:", c.get("role"), "| rule_family:", c.get("rule_family_id"))
        print("  context text:", c.get("text", "")[:400])

from huggingface_hub import list_repo_files
print("\nrepo files:", list_repo_files("jhlee0619/mpib", repo_type="dataset"))

v1 = next(r for r in rows if r["vector"] == "V1")
print("\n=== full V1 row ===")
print(json.dumps(v1, indent=1)[:3000])

tr = hf_hub_download("jhlee0619/mpib", "data/train.jsonl", repo_type="dataset")
trows = [json.loads(l) for l in open(tr)]
print("\ntrain vectors:", Counter(r["vector"] for r in trows))
print("train redacted V2 contexts:", sum(1 for r in trows if r["vector"]=="V2" for c in r["contexts"] if c.get("text")=="[REDACTED_PAYLOAD]"))