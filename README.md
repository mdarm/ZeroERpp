# ZeroER++

Run from the repository root with Docker and Make installed.

1. Build the environment: `make init`.
2. Download the datasets and embeddings: `make run_module module="python main_zeroer.py download"`.

The download contains 975 files (~68 GB) from fixed dataset versions, including precomputed
numerical representations of text (embeddings).

Recreate blocking, unsupervised selection, and matching results:

```bash
make run_module module="python main_zeroer.py verify-data"
make run_module module="python main_zeroer.py prepare"
make run_module module="bash scripts/blocking.sh"
make run_module module="bash scripts/primary.sh"
make run_module module="bash scripts/sensitivity.sh"
make run_module module="bash scripts/baselines.sh"
make run_module module="python main_zeroer.py report"
```

`primary.sh` and `sensitivity.sh` include hyperparameter selection, full-data refits,
and matching evaluation. The report command exports the accuracy tables.

Measure runtimes after the results above, then verify the complete reproduction:

```bash
make run_module module="bash scripts/timings.sh"
make run_module module="bash scripts/feature_timings.sh"
make run_module module="bash scripts/report.sh"
```

The final report compares results and timing coverage with `results/reference/`.
To run both groups in one pass: `make run_module module="bash scripts/reproduce.sh"`.

Outputs are saved under `results/runs/`: experiment databases, generated features,
`reports/`, and `magellan_feature_timings.csv`. Completed work resumes with the same
code and inputs; after code changes, use a fresh `--output-dir`.

The experiments use five datasets and twenty embedding models. Matching is evaluated
on 22 candidate-pair sets (each defined by the dataset, the table supplying search
queries, and the number of nearest neighbours retained per record): five for the main
comparison and nineteen for sensitivity analysis, with two sets shared.

For each candidate set, we evaluate 27 feature configurations: twenty individual
embedding models and seven combinations. Hyperparameters are selected within each
feature configuration by minimising marginal validation negative log-likelihood
(NLL) across three record-disjoint train/validation splits. Reported model rankings
and exact threshold optima use ground-truth labels; fixed decision rules do not.

Timings use serial execution with one CPU thread. Each fitted matching method and
direct cosine-similarity baseline is benchmarked in three repetitions with the same
inputs; reports use the median runtime to reduce the effect of run-to-run variation.
The original string baselines are timed once on Fodors–Zagats `right/k1`. A separate
benchmark times fresh Magellan feature generation once for each of the 22 candidate
sets. Embedding generation and blocking are excluded from matching times.

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
