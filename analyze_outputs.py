"""Chapter 4: completeness of the runs and form of the model outputs, per model.

Usage: python analyze_outputs.py -> results/tables/t4_outputs.csv
"""
import glob, os

import pandas as pd

from bench import load_config


def main():
    cfg = load_config(); A = cfg["analysis"]
    runs = pd.concat([pd.read_json(f, lines=True) for f in sorted(glob.glob(os.path.join(cfg["paths"]["runs_dir"], "*_*.jsonl")))],
                     ignore_index=True)
    s = pd.read_json(os.path.join(cfg["paths"]["results_dir"], "scores.jsonl"), lines=True)
    assert runs.run_id.is_unique and set(runs.run_id) == set(s.run_id)
    runs["words"] = runs.output.fillna("").astype(str).str.split().str.len()
    runs["no_output"] = runs.error.notna()
    g, gs = runs.groupby("model"), s.groupby("model")
    gen, att = s[s.type == "genuine"].groupby("model"), s[s.type == "attack"].groupby("model")
    t = pd.DataFrame({
        "runs": g.size(), "no output": g.no_output.sum(), "median words": g.words.median(),
        "mean latency s": g.latency_s.mean().round(1),
        "attack outputs letter only": att.judge_model.apply(lambda x: (x == "letter_only").sum()),
        "genuine: letter stated": gen.letter_source.apply(lambda x: (x == "letter").sum()),
        "genuine: letter from option text": gen.letter_source.apply(lambda x: (x == "option_text").sum()),
        "genuine: no answer": gen.no_answer.sum().astype(int), "document echo": gs.doc_echo.sum()})
    t = t.loc[list(cfg["models"])]; t.index = [A["model_labels"][m] for m in t.index]
    t.loc["Total"] = t.sum(numeric_only=True); t.loc["Total", ["median words", "mean latency s"]] = [runs.words.median(), round(runs.latency_s.mean(), 1)]
    t.to_csv(os.path.join(A["tables_dir"], "t4_outputs.csv")); print(t.to_string())
    print(runs.groupby(["model", "condition"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
