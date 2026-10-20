#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Edit the JSON grid; add --prepare on the first run to generate inputs.
# Preparation and fitting use no match labels.
# Add --evaluate when gold-based diagnostics are wanted after fitting.
"${PYTHON:-python3}" -u main_zeroer.py tune \
    --config configs/unsupervised.json --stage all "$@"
