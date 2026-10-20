# Generate Magellan features

Run from the repository root. Supply the original benchmark tables and a candidate-pair CSV in this layout:

```text
data/benchmarks/fodors_zagats/left.csv
data/benchmarks/fodors_zagats/right.csv
results/runs/ladder_pairs/fodors_zagats__right_k4.csv
```

The tables need a unique `id` column. The candidate CSV needs `ltable_id` and `rtable_id` columns containing IDs from those tables, in the order used by the experiment. Extraction requires neither embeddings nor `matches.csv`.

Build the dependency image:

```bash
make init
```

Generate the string features using the existing feature API:

```bash
make run_module module="python -" <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

from data.reference import load_reference_tables
from data.features.strings import build_zeroer_features

dataset, side, k = "fodors_zagats", "right", 4
name = f"{dataset}__{side}_k{k}"
left, right = load_reference_tables(dataset, "data/benchmarks")
ids = pd.read_csv(
    Path("results/runs/ladder_pairs") / f"{name}.csv", dtype=str
)[["ltable_id", "rtable_id"]]

left_rows = pd.Series(np.arange(len(left)), index=left["id"].astype(str))
right_rows = pd.Series(np.arange(len(right)), index=right["id"].astype(str))
pairs = np.column_stack([
    left_rows.loc[ids["ltable_id"]].to_numpy(),
    len(left) + right_rows.loc[ids["rtable_id"]].to_numpy(),
])
entities = pd.concat([left.astype(object), right.astype(object)], ignore_index=True)
attributes = [column for column in left if column != "id" and column in right]
features = build_zeroer_features(
    pairs, entities, attributes, dataset_limit=len(left),
    drop_zero_variance=False, n_jobs=4,
)

output = Path("results/runs/magellan-features") / f"{name}.csv.gz"
if output.exists() or Path(str(output) + ".json").exists():
    raise FileExistsError(f"Choose a fresh output path: {output}")
output.parent.mkdir(parents=True, exist_ok=True)
pd.concat([ids, features], axis=1).to_csv(output, index=False, compression="gzip")
print(f"Wrote {output}: {len(ids)} pairs, {features.shape[1]} features")
PY
```

Change `dataset`, `side`, and `k` to match the supplied candidate file. The output contains both candidate IDs followed by the string-similarity columns, preserving candidate order. `drop_zero_variance=False` keeps the full feature schema; the baseline loader removes constants and excluded attributes before fitting.

For a local Python 3.10 environment, run `python -m pip install -r requirements.txt` and replace `make run_module module="python -"` with `python -` in the command above.
