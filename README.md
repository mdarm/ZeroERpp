# ZeroER++

Run from the repository root with Docker and Make installed.

1. Build the environment: `make init`.
2. Download the datasets and embeddings: `make run_module module="python main_zeroer.py download"`.
3. Reproduce the results: `make run_module module="bash scripts/reproduce.sh"`.

The download contains 975 files (~68 GB) from fixed dataset versions, including precomputed
numerical representations of text (embeddings). `reproduce.sh` checks the inputs,
selects possible record pairs to compare (blocking), computes similarity measurements
(features), runs the matching methods, measures runtimes, and compares results with
`results/reference/`.

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

The experiments use five datasets and twenty text-embedding models. Matching is
tested on 22 sets of possible record pairs: five for the main comparison and nineteen
to check the effect of changing which pairs are compared (two sets overlap). Each set
specifies a dataset, which table supplies the search queries, and how many nearest
neighbours are kept per record.

For each set, 27 ways of constructing similarity features are compared: twenty
individual embedding models and seven combinations. Within each choice, settings are chosen by
minimising negative log-likelihood (NLL), which measures how poorly the model explains
the validation pairs, without consulting known correct matches. Training and
validation share no records within any of the three splits. Reported model rankings
and best-possible score thresholds use known correct matches; applying predefined
decision rules does not.

Measurements run one at a time on one CPU thread. Fitted matching methods and direct
cosine-similarity comparisons are timed three times; the original string-based methods
are timed once. Magellan string features are also computed afresh once for each of
the 22 pair sets. Embedding creation and candidate-pair selection are excluded from
matching times.

Verification compares match counts, chosen settings, and accuracy against the saved
results. It also checks timing coverage for 593 successful model fits, direct cosine
comparisons for each of the 22 pair sets, and two original string-based methods (617 in total).
Runtimes may differ by machine.

Some failed fits are expected, including DBLP–ACM when each record in the right table keeps
twelve nearest neighbours (`right/k12`) and all twenty embedding models contribute
separate features (`all:separate`). These failures are retained in the saved results.

The completed rerun reproduces all matching results. Two ABT–Buy results for selecting
candidate pairs differ by one known correct pair: `gte_large` with two neighbours per
record in the right table (`right/k2`), and `st5_xl` with eight (`right/k8`). Strict comparison
returns a failure exit status for these differences, after saving the tables and
`reports/report.json`.

Infrastructure used:

- Original experiments: AWS EC2 `g5.4xlarge` (16 vCPUs, 64 GB RAM, NVIDIA A10G 24 GB)
  and `r6i.8xlarge` (32 vCPUs, 256 GB RAM), both running Ubuntu 20.04.6 LTS.
- Embedding creation: KVM VM with AMD EPYC 9B45 (48 vCPUs), 181 GB RAM, and
  NVIDIA RTX PRO 6000 Blackwell with 96 GB VRAM.
- All stages except embedding creation were also run on this laptop: Lenovo
  ThinkPad T16 Gen 3, Intel Core Ultra 5 135U (12 cores, 14 threads), 32 GB RAM,
  Ubuntu 22.04.5 LTS. The Docker environment uses Python 3.10 and fixed dependency versions.
