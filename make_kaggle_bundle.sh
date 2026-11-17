#!/usr/bin/env bash
# Pack what run_experiment.py needs on Kaggle into one zip (code, config, benchmark, retrieval cache).
# Upload kaggle_bundle/thesis_bundle.zip as a PRIVATE Kaggle dataset (it contains MPIB attack text).
set -euo pipefail
cd "$(dirname "$0")"
rm -rf kaggle_bundle && mkdir -p kaggle_bundle/thesis/data
cp bench.py defenses.py rag.py run_experiment.py config.yaml kaggle_bundle/thesis/
cp -r data/benchmark kaggle_bundle/thesis/data/
cp data/retrieval.jsonl kaggle_bundle/thesis/data/
cp kaggle_run.ipynb kaggle_bundle/
(cd kaggle_bundle && zip -qr thesis_bundle.zip thesis && rm -rf thesis)
ls -la kaggle_bundle
echo "Upload kaggle_bundle/thesis_bundle.zip as a private dataset named 'thesis-bundle'; open kaggle_run.ipynb as a new notebook."
