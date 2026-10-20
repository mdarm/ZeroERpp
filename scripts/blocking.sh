#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# 5 datasets x 20 frozen record encoders x 2 directions x k=1..100.
exec "${PYTHON:-python3}" -u main_zeroer.py blocking --k-max 100 --directions right left "$@"
