"""Step 5/6: run model x condition x item through the simulated RAG system and log every run.

Usage:
  python run_experiment.py --model llama3 --condition baseline --limit 60      # pilot
  python run_experiment.py --model llama3 --condition all                      # one model, all conditions
  python run_experiment.py --model all --condition all                         # everything (overnight)

Each run appends one JSON line to runs/<model>_<condition>.jsonl. Resume-safe: existing run_ids are skipped.
Genuine items: clean document, one run. Attack items: one run per hiding technique allowed for the condition
(config.techniques_by_condition). Condition medft is an analysis alias (baseline on the medical models) and
is skipped here.
"""
import argparse, hashlib, json, os, time
from bench import load_config, load_items
from defenses import BUILDERS

_CACHE = None
def cached_retrieval(cfg, item, variant, sanitize, all_ids):
    """Chunks + retrieval info from data/retrieval.jsonl (precompute_retrieval.py). Live fallback if absent."""
    global _CACHE
    path = cfg["paths"].get("retrieval_cache", "data/retrieval.jsonl")
    if _CACHE is None:
        _CACHE = {}
        if os.path.exists(path):
            for l in open(path):
                if l.strip():
                    r = json.loads(l); _CACHE[r["key"]] = r
        print(f"[run] retrieval cache: {len(_CACHE)} entries from {path}")
    k = f"{item['id']}|{variant}|{'san' if sanitize else 'raw'}"
    r = _CACHE.get(k)
    if r is None:
        from rag import retrieve
        print(f"  [run] cache miss {k} - retrieving live (run precompute_retrieval.py to avoid this)")
        return retrieve(cfg, item, variant, all_ids, item["question"], sanitize=sanitize)
    info = {x: r[x] for x in ("n_corpus_chunks", "n_target_chunks_retrieved", "retrieved_contains_injection",
                               "sanitize_spans_dropped", "injection_overlap", "best_target_sim", "best_distractor_sim")}
    return r["chunks"], info

def ollama_chat(cfg, model_tag, system, user):
    if os.environ.get("FAKE_OLLAMA"):
        return f"FAKE[{model_tag}] B. (echo: {user[-80:]})"
    import ollama
    g = cfg["generation"]
    if model_tag in {cfg["models"][k] for k in cfg.get("completion_models", [])}:
        # base model: plain completion prompt, no chat template (see config.yaml completion_models)
        r = ollama.generate(model=model_tag, keep_alive=g["keep_alive"], raw=True,
                            prompt=system + "\n\n" + user + cfg["answer_cue"],
                            options={"temperature": g["temperature"], "seed": g["seed"],
                                     "num_predict": g["num_predict"], "num_ctx": g["num_ctx"]})
        return r["response"]
    r = ollama.chat(model=model_tag, keep_alive=g["keep_alive"],
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    options={"temperature": g["temperature"], "seed": g["seed"],
                             "num_predict": g["num_predict"], "num_ctx": g["num_ctx"]})
    return r["message"]["content"]

def techniques_for(cfg, condition):
    t = cfg["techniques_by_condition"]
    return t.get(condition, t["default"])

def planned_runs(cfg, items, model_key, condition):
    for it in items:
        variants = ["clean"] if it["type"] == "genuine" else techniques_for(cfg, condition)
        for v in variants:
            yield it, v, f"{model_key}|{condition}|{it['id']}|{v}"

def load_done(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["run_id"] for l in open(path) if l.strip()}

def run(cfg, items, model_key, condition, limit=None, all_ids=None):
    if condition == "medft":
        print("[run] medft is an analysis alias of baseline on", cfg["medft_models"], "- skipping"); return
    model_tag = cfg["models"][model_key]
    os.makedirs(cfg["paths"]["runs_dir"], exist_ok=True)
    out_path = os.path.join(cfg["paths"]["runs_dir"], f"{model_key}_{condition}.jsonl")
    done = load_done(out_path)
    all_ids = all_ids or [it["id"] for it in items]   # distractors always come from the full benchmark
    build = BUILDERS[condition]
    todo = [(it, v, rid) for it, v, rid in planned_runs(cfg, items, model_key, condition) if rid not in done]
    if limit:
        todo = todo[:limit]
    print(f"[run] {model_key}/{condition}: {len(done)} done, {len(todo)} to do -> {out_path}")
    t_start = time.time()
    consecutive_errors = 0
    with open(out_path, "a") as f:
        for n, (it, variant, rid) in enumerate(todo, 1):
            t0 = time.time()
            chunks, rinfo = cached_retrieval(cfg, it, variant, condition == "sanitize", all_ids)
            system, user, dinfo = build(chunks, it)
            try:
                output, err = ollama_chat(cfg, model_tag, system, user), None
                consecutive_errors = 0
            except Exception as e:
                output, err = "", repr(e)
                if "repeat limit" in err:    # Ollama aborted a degenerate, looping generation (base models in raw
                    err = "degenerate:" + err  # mode): that is a model outcome, logged as an empty answer, not a failure
                    consecutive_errors = 0
                    print(f"  DEGENERATE {rid}: {err[:120]}", flush=True)
                else:                        # print it, count it, stop if Ollama is clearly down
                    consecutive_errors += 1
                    print(f"  ERROR on {rid}: {err[:300]}", flush=True)
                    if consecutive_errors >= 3:
                        raise SystemExit("[run] 3 consecutive errors - is Ollama running? (ollama ps)")
            row = {"run_id": rid, "item_id": it["id"], "source": it["source"], "type": it["type"],
                   "category": it.get("category"), "technique": variant, "model": model_key, "model_tag": model_tag,
                   "condition": condition, "prompt_sha256": hashlib.sha256((system + user).encode()).hexdigest(),
                   "prompt_chars": len(system) + len(user), "output": output, "error": err,
                   "latency_s": round(time.time() - t0, 2), "answer_idx": it.get("answer_idx"),
                   "injected_instruction": it.get("injected_instruction"), "target_phrase": it.get("target_phrase"),
                   **rinfo, **dinfo}
            if err is None or err.startswith("degenerate:"):
                f.write(json.dumps(row) + "\n"); f.flush()
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t_start
                print(f"  {n}/{len(todo)}  avg {el/n:.1f}s/run  eta {(len(todo)-n)*el/n/60:.0f} min", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model key from config, or 'all'")
    ap.add_argument("--condition", required=True, help="condition from config, or 'all'")
    ap.add_argument("--limit", type=int, default=None, help="max new runs")
    ap.add_argument("--pilot", type=int, default=None, help="use a seeded stratified subset of N items (N/3 per source)")
    a = ap.parse_args()
    cfg = load_config()
    items = load_items(cfg, require_advbench=True)
    all_ids = [it["id"] for it in items]
    if a.pilot:
        import random
        rng = random.Random(cfg["seed"])
        per = a.pilot // 3
        sub = []
        for src in ("MedQA", "MPIB", "AdvBench"):
            pool = [it for it in items if it["source"] == src]
            sub += rng.sample(pool, min(per, len(pool)))
        items = sub
        print(f"[run] pilot: {len(items)} items ({per} per source)")
    models = list(cfg["models"]) if a.model == "all" else [a.model]
    conds = [c for c in cfg["conditions"] if c != "medft"] if a.condition == "all" else [a.condition]
    for m in models:
        for c in conds:
            run(cfg, items, m, c, a.limit, all_ids=all_ids)
