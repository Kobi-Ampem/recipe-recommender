#!/usr/bin/env bash
# Rebuild every number in the README and docs/ from the raw data.
# Needs data/foodcom/ (see README); MovieLens downloads itself.
# Takes about 1.5 h on a laptop CPU, almost all of it the BPR grid on Food.com.
set -euo pipefail
uv sync
uv run python -m reciperec.profile
for dataset in ml-1m foodcom; do
  uv run python -m reciperec.tune --dataset "$dataset"
  uv run python -m reciperec.evaluate --dataset "$dataset" \
    --model popularity itemknn puresvd bpr --tuned
done
uv run python -m reciperec.report
