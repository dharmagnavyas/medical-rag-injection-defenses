"""Shared loaders. Every script reads settings from config.yaml through here."""
import csv, json, os, random, yaml

def load_config(path="config.yaml"):
    return yaml.safe_load(open(path))

def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def load_items(cfg, require_advbench=False):
    """Return the 600 benchmark items with a uniform shape.

    Every item has: id, source, type ('genuine'|'attack'), question.
    Attack items add injected_instruction (and target_phrase for AdvBench).
    AdvBench rows have no clinical question of their own, so each is paired with a MedQA vignette
    (carrier_medqa_id, chosen with the config seed) that supplies the user query and the document body.
    """
    p = cfg["paths"]
    items = read_jsonl(p["medqa"])
    medqa_by_id = {it["id"]: it for it in items}

    if os.path.exists(p["mpib"]):
        items += read_jsonl(p["mpib"])
    else:
        print(f"[bench] {p['mpib']} missing; run build_mpib.py (MPIB payloads are not committed)")

    adv_path = p["advbench"] if os.path.exists(p["advbench"]) else p["advbench"].replace(".csv", "_TO_FILL.csv")
    rows = list(csv.DictReader(open(adv_path)))
    rng = random.Random(cfg["seed"])
    carriers = rng.sample(list(medqa_by_id), len(rows))
    n_filled = 0
    for r, cid in zip(rows, carriers):
        if not r["medical_injection"].strip():
            if require_advbench:
                raise SystemExit(f"[bench] {r['id']} has no medical_injection in {adv_path}")
            continue
        n_filled += 1
        items.append({
            "id": r["id"], "source": "AdvBench", "type": "attack",
            "category": r["harm_category"], "goal": r["goal"],
            "carrier_medqa_id": cid, "question": medqa_by_id[cid]["question"],
            "options": medqa_by_id[cid]["options"], "answer_idx": medqa_by_id[cid]["answer_idx"],
            "injected_instruction": r["medical_injection"].strip(),
            "target_phrase": r["target_phrase"].strip(),
            "review_status": r.get("review_status", "draft"),
        })
    print(f"[bench] loaded {len(items)} items ({n_filled}/{len(rows)} AdvBench rows filled, from {adv_path})")
    return items
