#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# 19 frozen ladder cells x 27 panels; primary overlaps resume without duplicate fits.
exec "${PYTHON:-python3}" -u main_zeroer.py match --scope sensitivity --workers 4 "$@"
