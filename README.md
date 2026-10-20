# ZeroER++

Run from the repository root with Docker and Make installed.

1. Build the environment: `make init`.
2. Download the datasets and embeddings: `make run_module module="python main_zeroer.py download"`.
3. Reproduce the results: `make run_module module="bash scripts/reproduce.sh"`.

Features are generated automatically. Outputs are saved under `results/runs/`.

For hyperparameter tuning, edit `configs/unsupervised.json` and run:

```bash
make run_module module="bash scripts/tune.sh --prepare"
```

For timing measurements and updated reports:

```bash
make run_module module="bash scripts/timings.sh"
make run_module module="bash scripts/report.sh"
```
