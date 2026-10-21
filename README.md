# ZeroER++

Run from the repository root with Docker and Make installed.

1. Build the environment: `make init`.
2. Download the datasets and embeddings: `make run_module module="python main_zeroer.py download"`.
3. Reproduce the results: `make run_module module="bash scripts/reproduce.sh"`.

The download contains 975 pinned files (~68 GB), including precomputed embeddings.
`reproduce.sh` verifies inputs, generates features, runs blocking and matching,
fits the baselines, measures timings, and compares reports with `results/reference/`.

To run the stages separately, in order:

```bash
make run_module module="python main_zeroer.py verify-data"
make run_module module="python main_zeroer.py prepare"
make run_module module="bash scripts/blocking.sh"
make run_module module="bash scripts/primary.sh"
make run_module module="bash scripts/sensitivity.sh"
make run_module module="bash scripts/baselines.sh"
make run_module module="bash scripts/timings.sh"
make run_module module="bash scripts/feature_timings.sh"
make run_module module="bash scripts/report.sh"
```

Outputs are saved under `results/runs/`: experiment databases, generated features,
`reports/`, and `magellan_feature_timings.csv`. Completed work resumes with the same
code and inputs; after code changes, use a fresh `--output-dir`.

The frozen protocol covers five datasets, twenty encoders, five primary cells and
nineteen sensitivity cells (22 distinct cells), and 27 matching panels. Configuration
selection uses three-fold record-disjoint validation NLL within each panel. Model
rankings and threshold oracles use gold labels; fixed decision rules do not.

Timings are serial: three repeats for fitted and cosine matching, one for the
published string baselines, and one fresh Magellan feature pass across all 22 cells.
Embedding generation and blocking are excluded from matching timings.

Verification checks counts, selected configurations, metrics, and 617 timing groups;
wall-clock times may differ. Failed search configurations and the DBLP–ACM right/k12
`all:separate` collapsed-mixture refit are expected reference outcomes.

The completed replay matches all matching references. Two historical ABT–Buy blocking
rows differ by one gold pair: `gte_large` right/k2 and `st5_xl` right/k8. Strict comparison
still exits nonzero for those differences, after saving the tables and `reports/report.json`.

Infrastructure used:

- Original experiments: AWS EC2 `g5.4xlarge` (16 vCPUs, 64 GB RAM, NVIDIA A10G 24 GB)
  and `r6i.8xlarge` (32 vCPUs, 256 GB RAM), both running Ubuntu 20.04.6 LTS.
- Embedding creation: KVM VM with AMD EPYC 9B45 (48 vCPUs), 181 GB RAM, and
  NVIDIA RTX PRO 6000 Blackwell with 96 GB VRAM.
- All stages except embedding creation were also run on this laptop: Lenovo
  ThinkPad T16 Gen 3, Intel Core Ultra 5 135U (12 cores, 14 threads), 32 GB RAM,
  Ubuntu 22.04.5 LTS. The Docker environment uses Python 3.10 and pinned dependencies.
