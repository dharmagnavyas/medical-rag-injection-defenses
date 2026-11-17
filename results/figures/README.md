# Figure guide (results/figures)

All figures are produced by `analyze.py` from `results/scores.jsonl` at 300 dpi. Each title states the takeaway; the
subtitle states what was counted.

| File | What it shows | Pointer for the reader | Suggested place |
|---|---|---|---|
| fig_asr_all.png | ASR per model x defense, all 300 attack items, 95% CIs | Only the orange (sanitization) bar drops; the other three defenses sit inside the no-defense CI. Values on the no-defense and sanitized bars. | 5.3 ASR by defense x model (main result) |
| fig_asr_mpib.png | Same, MPIB items only (planted clinical answers) | Without defense 68-89% of planted answers are adopted; sanitization leaves 11-33%. | 5.3, second panel or appendix |
| fig_asr_by_source.png | No-defense ASR split by attack source | MPIB 68-89% vs AdvBench 0-3% although the poisoned chunk was retrieved in 95% of runs for both: alignment training blocks overt harm, not planted clinical content. | 5.2 Baseline |
| fig_asr_heatmap.png | ASR grid, model x defense | The sanitization column is boxed: it is the only column that changes. Meditron/StruQ (42) is document regurgitation, see doc_echo in T9. | 5.3 or 5.12 summary |
| fig_utility.png | MedQA accuracy per model x defense | Flat across defenses for every model: no defense costs accuracy. Meditron is low because it is a base model answered in completion mode. | 5.4 FRR and utility |
| fig_tradeoff_utility.png | ASR vs MedQA accuracy, one point per model x defense | Two horizontal bands separated by the dashed line: sanitization moves every model down (fewer attacks) without moving it left (no accuracy loss). | 5.5 Trade-off (the key figure) |
| fig_tradeoff_frr.png | ASR vs false refusal rate | Every point sits at or below 0.3% FRR: no defense blocks genuine questions. Answers RQ3 (null result). | 5.4 or 5.10 RQ3 |
| fig_sanitize_by_technique.png | Sanitized ASR for white text, 1-pt font and metadata vs no defense | The three sanitized bars are equal within CI: the sanitizer is indifferent to how the text was hidden. | 5.3 or 5.8 RQ1 |
| fig_rouge_l_injection.png | Mean ROUGE-L F1 of the output against the injected text, MPIB rows | Graded companion to ASR: sanitization cuts copied text from ~0.35-0.45 to ~0.08-0.16; prompt defenses do not. | 5.6 Statistics or appendix |

Tables with the exact numbers and CIs are in `results/tables/` and, as markdown, in `results/analysis.md`
(T1 main grid, T4 McNemar/Holm, T5 bootstrap deltas, T6 medical vs general pairs, T8 retrieval-conditional ASR,
T9 judge diagnostics, T11 ROUGE-L).
