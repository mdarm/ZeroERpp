#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
"${PYTHON:-python3}" -u main_zeroer.py verify-data --scope all "$@"
"${PYTHON:-python3}" -u main_zeroer.py prepare --scope all "$@"
bash scripts/blocking.sh "$@"
"${PYTHON:-python3}" -u main_zeroer.py match --scope all --workers 4 "$@"
bash scripts/baselines.sh "$@"
bash scripts/timings.sh "$@"
bash scripts/feature_timings.sh "$@"
bash scripts/report.sh "$@"
