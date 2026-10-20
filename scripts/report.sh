#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${PYTHON:-python3}" -u main_zeroer.py report --compare-reference "$@"
