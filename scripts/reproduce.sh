#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Download the immutable inputs first: python main_zeroer.py download
# Pass shared --data-dir / --output-dir / --threads / --dry-run options here.
"${PYTHON:-python3}" -u main_zeroer.py prepare --scope all "$@"
bash scripts/blocking.sh "$@"
"${PYTHON:-python3}" -u main_zeroer.py match --scope all --workers 4 "$@"
bash scripts/baselines.sh "$@"
bash scripts/report.sh "$@"
# Run scripts/timings.sh separately for the serial timing measurements, then
# scripts/report.sh again to include the fresh timing tables in the report.
