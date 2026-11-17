# Protecting Medical RAG Systems from Indirect Prompt Injection

Code, benchmark files, run logs and results for the MSc thesis *Protecting Medical Retrieval-Augmented Generation
(RAG) Systems from Indirect Prompt Injection: A Comparative Study of Sanitization and Structural Defenses*
(Dharmagna Vyas, Liverpool John Moores University, 2026).

The study hides an instruction inside a synthetic discharge summary (white text, 1-pt font or PDF metadata), lets a
small RAG pipeline retrieve it, and measures how often six open models follow it with and without a defense.

## Design

| Part | Details |
|---|---|
| Benchmark | 600 items: 300 genuine MedQA questions, 150 MPIB V1 attacks, 150 AdvBench goals adapted to medicine |
| Models (Ollama) | `llama3:8b`, `mistral:7b`, `qwen2.5:7b`, `gemma:7b`, `meditron:7b`, `hf.co/BioMistral/BioMistral-7B-GGUF:Q4_K_M` |
| Conditions | no defense, visual sanitization, Spotlighting, StruQ-style prompting, MAP-style four-stage filter; medical fine-tuning is the no-defense comparison of Meditron and BioMistral with their general counterparts |
| Retrieval | `all-MiniLM-L6-v2` embeddings, FAISS, top-k 3 |
| Generation | temperature 0, seed 42, at most 256 new tokens |
| Metrics | attack success rate (ASR), false refusal rate (FRR), utility (MedQA answer letter correct) |
| Runs | 21,600: per model 300 genuine x 5 conditions, 300 attacks x 4 conditions, and 300 attacks x 3 hiding techniques under sanitization |

Every setting is in `config.yaml`. Model details are in `models.md`.

## What is in the repository

| Path | Contents |
|---|---|
| `build_medqa.py`, `build_mpib.py`, `build_advbench.py` | build the three parts of the benchmark |
| `build_pdfs.py` | writes one clean PDF per item and the attacked variants |
| `precompute_retrieval.py`, `rag.py` | chunking, embedding and retrieval |
| `defenses.py` | the five prompt and document conditions |
| `run_experiment.py` | runs model x condition x item and appends to `runs/<model>_<condition>.jsonl` |
| `score.py` | rule-based scoring plus an LLM judge (`qwen2.5:7b`; `mistral:7b` for Qwen's own outputs) |
| `analyze.py`, `analyze_*.py`, `make_diagrams.py` | tables, statistical tests and figures |
| `kaggle_run.ipynb`, `kaggle_score.ipynb`, `make_kaggle_bundle.sh` | how the full runs were executed on a Kaggle T4 GPU |
| `data/benchmark/` | `medqa_300.jsonl`, `advbench_150.csv`, `mpib_sample_ids.txt` |
| `runs/` | the 21,600 run logs, one JSON line per run |
| `runs/invalid_meditron_chatml/` | the first Meditron runs, discarded because the model echoed its chat template (thesis Chapter 3) |
| `results/scores.jsonl` | one score row per run |
| `results/analysis.md`, `results/tables/`, `results/figures/` | every table and figure used in the thesis |

## What is withheld

The MPIB dataset may not be redistributed, so this repository does not contain its attack text.

- `data/benchmark/mpib_150.jsonl` is not included. `build_mpib.py` rebuilds it from the Hugging Face dataset
  `jhlee0619/mpib` using the ids in `mpib_sample_ids.txt`.
- In `runs/`, the 7,350 rows of MPIB items carry the marker `[withheld: MPIB terms of use; ...]` in place of
  `injected_instruction`, `output` and `mapfilter_removed`. The fields `output_sha256` and `output_chars` identify
  the original output. All other fields, and the scores of these rows in `results/scores.jsonl`, are complete.
- The generated PDFs (`data/docs/`) and the retrieval cache (`data/retrieval.jsonl`) are not included because they
  embed the same text. Both are rebuilt by the steps below.

Rows for MedQA and AdvBench items are complete.

## Reproducing

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3:8b            # and the other five model tags in config.yaml

python build_medqa.py            # already in data/benchmark/
python build_mpib.py             # needs access to the MPIB dataset on Hugging Face
python build_advbench.py         # already in data/benchmark/
python build_pdfs.py
python precompute_retrieval.py

python run_experiment.py --model llama3 --condition baseline --limit 60   # pilot
python run_experiment.py --model all --condition all                      # all 21,600 runs
python score.py
python analyze.py
```

`run_experiment.py` and `score.py` are resume-safe: runs already logged are skipped.

`analyze.py` reads `results/scores.jsonl` and reproduces the tables and figures from the files in this repository,
with one exception: the ROUGE-L table and figure (T11) are computed from the full outputs. `score.py` and
`analyze_outputs.py` also read the full outputs. On the MPIB rows these three give the published numbers only
after the runs have been regenerated; run on the withheld rows they would overwrite T11 with wrong values.

The thesis runs used Ollama 0.40.16 on a Kaggle T4 GPU.

## Data sources

- MedQA (Jin et al., 2021), USMLE four-option questions.
- MPIB (Lee, Jang and Choi, 2026), Hugging Face dataset `jhlee0619/mpib`, V1 test split.
- AdvBench harmful behaviours (Zou et al., 2023); 150 goals were rewritten as medical instructions for this study.

The discharge summaries are synthetic. No real patient data was used.
