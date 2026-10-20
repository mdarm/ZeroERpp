#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Serial matching costs: 3 repeats per successful selected fit and direct-cosine
# cell; 1 fresh-string-feature repeat for Fodors-Zagats right/k1 only.
exec "${PYTHON:-python3}" -u main_zeroer.py timings --scope all --kind published \
  --repeats 3 --string-repeats 1 --threads 1 "$@"
