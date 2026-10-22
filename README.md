# ZeroER++

Publication code for blocking, unsupervised selection, and matching across five
datasets, twenty embedding models, and 22 candidate sets.

## Run the experiments

Run from the repository root with Docker and Make installed. The download contains
975 files (~68 GB), including the frozen embeddings.

```bash
make init
make run_module module="python main_zeroer.py download"
make run_module module="bash scripts/reproduce.sh"
```

This verifies inputs, prepares features, runs blocking, hyperparameter selection,
matching and timings, then exports and checks the publication results. Outputs are
saved in `results/runs/`. Individual stages are available in `scripts/`.

## Rerun the Optuna studies

Each study evaluates ten configurations using three record-disjoint splits and
validation negative log-likelihood. The selected configuration is refitted on all
candidates. The grid is defined in [exp/protocol.py](exp/protocol.py).

The same output directory resumes completed work. To repeat the studies, use a new
directory for both preparation and matching:

```bash
make run_module module="python main_zeroer.py prepare --output-dir results/rerun"
make run_module module="python main_zeroer.py match --workers 4 --output-dir results/rerun"
```

## Regenerate embeddings (optional)

In a separate Python environment with suitable GPUs:

```bash
python -m pip install -r requirements.txt -r requirements-encoding.txt
python -m data.encoding --data-dir data/benchmarks --output-dir results/regenerated-inputs
```

The output directory must not exist. Use `--data-dir results/regenerated-inputs`
and a fresh `--output-dir` for subsequent preparation and matching. Historical
checkpoint versions were not pinned, so new vectors may differ; `verify-data`
checks the frozen inputs. Original recipes are in `scripts/encoding/source/`.

## Verification

Reports check matching results, all 5,940 search configurations, publication
diagnostics, and timing coverage against the original artifacts. Expected failed
fits are retained. Two known ABT–Buy blocking-count differences cause strict
verification to exit with a failure status after saving `reports/report.json`.
All 22 candidate sets used for matching agree with the reference.

Run the tests with
`make run_module module="python -m unittest discover -s tests"`.

## Infrastructure

- Original experiments: AWS EC2 `g5.4xlarge` (16 vCPUs, 64 GB RAM, NVIDIA A10G 24 GB)
  and `r6i.8xlarge` (32 vCPUs, 256 GB RAM), both running Ubuntu 20.04.6 LTS.
- Embedding creation: KVM VM with AMD EPYC 9B45 (48 vCPUs), 181 GB RAM, and
  NVIDIA RTX PRO 6000 Blackwell with 96 GB VRAM.
- All stages except embedding creation were also successfully run on a Lenovo
  ThinkPad T16 Gen 3, Intel Core Ultra 5 135U (12 cores, 14 threads), 32 GB RAM,
  Ubuntu 22.04.5 LTS. The Docker environment uses Python 3.10 and fixed dependency versions.
