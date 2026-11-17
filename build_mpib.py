import json, random, re, os
from collections import Counter, defaultdict
from huggingface_hub import hf_hub_download

SEED, N = 42, 150
os.makedirs("data/benchmark", exist_ok=True)

rows = []
for split in ["train", "validation", "test"]:
    p = hf_hub_download("jhlee0619/mpib", f"data/{split}.jsonl", repo_type="dataset")
    rows += [json.loads(l) for l in open(p)]

v1 = [r for r in rows if r["vector"] == "V1" and "[INSTRUCTION]" in r["user_query"]]
print("V1 rows with visible instruction:", len(v1))

by_harm = defaultdict(list)
for r in v1:
    h = r["labels"]["harm_types"][0] if r["labels"]["harm_types"] else "H0"
    by_harm[h].append(r)
print("per harm type:", {k: len(v) for k, v in by_harm.items()})

random.seed(SEED)
quota = {h: round(N * len(v) / len(v1)) for h, v in by_harm.items()}
picked = []
for h, rs in by_harm.items():
    random.shuffle(rs)
    picked += rs[:quota[h]]
picked = picked[:N]

items = []
for n, r in enumerate(picked):
    q, inj = r["user_query"].split("[INSTRUCTION]", 1)
    items.append({
        "id": f"MPIB_{n:03d}",
        "source": "MPIB",
        "source_id": r["sample_id"],
        "type": "attack",
        "category": r["labels"]["harm_types"][0] if r["labels"]["harm_types"] else "H0",
        "rule": r["rule"],
        "scenario": r["scenario"],
        "severity": r["labels"]["severity"],
        "question": q.strip().strip('"'),
        "injected_instruction": inj.strip(),
        "expected_safe_behaviour": r["labels"]["expected_safe_behavior"],
    })

with open("data/benchmark/mpib_150.jsonl", "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")

print("saved", len(items))
print("categories:", Counter(it["category"] for it in items))
print("rules:", Counter(it["rule"] for it in items))
for it in random.sample(items, 5):
    print(f"\n{it['id']} [{it['category']}/{it['rule']}] -> {it['injected_instruction']}")