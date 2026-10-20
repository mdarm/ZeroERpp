#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# B-paper and B-code, on the same 22 union cells, original last-E-step posterior.
exec "${PYTHON:-python3}" -u main_zeroer.py baselines --scope all "$@"
