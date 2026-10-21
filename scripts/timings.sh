#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${PYTHON:-python3}" -u main_zeroer.py timings --scope all --kind published \
  --repeats 3 --string-repeats 1 --threads 1 "$@"
