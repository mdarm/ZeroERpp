#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# 5 KaLM cells x 27 panels; 6 initial + 4 refinement configs, 3 folds, seed 0.
# The CLI includes selected refits and MiniLM6 zero-fill/hard-EM ablations.
exec "${PYTHON:-python3}" -u main_zeroer.py match --scope primary --workers 4 "$@"
