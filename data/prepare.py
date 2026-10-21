"""Rebuild candidate pairs, string features, and similarities from raw inputs.

No labels or historical feature caches are read during preparation. Outputs are
bound to source contents and implementation; stale generated caches are rejected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp import config
from exp.protocol import MODELS, select_cells
from data.benchmarks import DATASETS

IDS = ["ltable_id", "rtable_id"]
SCHEMA = "zeroerpp-prepared-inputs-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sidecar(path):
    return Path(str(path) + ".json")


def _code_hash(*modules):
    files = [Path(__file__), *[Path(module.__file__) for module in modules]]
    return hashlib.sha256("\n".join(sha256_file(p) for p in files).encode()).hexdigest()


def _sources(paths, base):
    return {str(Path(p).relative_to(base)): sha256_file(p) for p in paths}


def _reusable(path, expected):
    path = Path(path)
    sidecar = _sidecar(path)
    if not path.exists():
        return False
    if not sidecar.exists():
        raise ValueError("Unverified generated output {}; use a fresh --output-dir".format(path))
    metadata = json.loads(sidecar.read_text())
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Sources or preparation code changed for {}; use a fresh --output-dir".format(path))
    if metadata.get("sha256") != sha256_file(path):
        raise ValueError("Generated output was modified: " + str(path))
    return True


def _write_frame(frame, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        compression = {"method": "gzip", "mtime": 0} if str(path).endswith(".gz") else None
        frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)
    _json(_sidecar(path), dict(metadata, sha256=sha256_file(path)))


def _base_metadata(dataset, side, k, kind):
    return dict(schema=SCHEMA, kind=kind, dataset=dataset, side=side, k=int(k))


def _tables(dataset):
    tables = [pd.read_csv(Path(config.BENCH_DIR) / dataset / (side + ".csv"),
                          usecols=["id"], dtype=str)["id"] for side in ("left", "right")]
    for table in tables:
        if table.isna().any() or table.duplicated().any():
            raise ValueError("Table IDs must be nonmissing and unique: " + dataset)
    return [table.to_numpy() for table in tables]


def _preserve_identical_prefix(neighbors, prefix, indexed_vectors):
    """Preserve k=2 order when FAISS reverses identical vectors at k=4."""
    neighbors = neighbors.copy()
    for row in np.unique(np.where(neighbors[:, :prefix.shape[1]] != prefix)[0]):
        for rank, expected in enumerate(prefix[row]):
            actual = neighbors[row, rank]
            if actual == expected:
                continue
            positions = np.flatnonzero(neighbors[row] == expected)
            if not len(positions) or not np.array_equal(indexed_vectors[actual], indexed_vectors[expected]):
                raise ValueError("Non-identical-vector prefix difference at row {}, rank {}".format(row, rank))
            position = positions[0]
            neighbors[row, rank], neighbors[row, position] = expected, actual
    return neighbors


def _retrieve(dataset, side, k, tables):
    from exp import blocking
    vectors = [np.load(Path(config.RECORD_DIR) / dataset / (config.MODEL + "_" + table + ".npy"),
                       mmap_mode="r", allow_pickle=False) for table in ("left", "right")]
    if any(len(vectors[i]) != len(tables[i]) for i in (0, 1)):
        raise ValueError("Record vectors and table IDs have different row counts: " + dataset)
    index, query = (vectors[1], vectors[0]) if side == "left" else (vectors[0], vectors[1])
    if k > len(index):
        raise ValueError("k exceeds the indexed table size")
    neighbors = blocking.nn_search(index, query, int(k), config.DISTANCE)
    if (dataset, side, k) == ("amazon_googleproducts", "left", 4):
        prefix = blocking.nn_search(index, query, 2, config.DISTANCE)
        neighbors = _preserve_identical_prefix(neighbors, prefix, index)
    queried = np.repeat(np.arange(len(query)), k)
    li, ri = (queried, neighbors.ravel()) if side == "left" else (neighbors.ravel(), queried)
    return pd.DataFrame({IDS[0]: tables[0][li], IDS[1]: tables[1][ri]}), li, ri


def _pair_source(dataset, side, k):
    largest = max(size for queried, size in config.LADDER[dataset] if queried == side)
    return max(largest, k)


def load_pairs(dataset, side, k, pairs_dir=None):
    """Read candidates in the exact order used by feature-bank slicing."""
    path = Path(pairs_dir or config.PAIRS_DIR) / (config.cell_name(dataset, side, k) + ".csv")
    if not path.exists():
        raise FileNotFoundError(str(path) + "; run python main_zeroer.py prepare first")
    if _sidecar(path).exists():
        metadata = json.loads(_sidecar(path).read_text())
        if metadata.get("sha256") != sha256_file(path):
            raise ValueError("Candidate file was modified: " + str(path))
    frame = pd.read_csv(path, dtype=str)
    if list(frame) != IDS or frame.isna().any().any() or frame.duplicated().any():
        raise ValueError("Invalid candidate IDs in " + str(path))
    return frame


def validate_feature_cache(source, dataset, side, source_k):
    """Validate generated provenance, or identify content-hashed external input.

    Callers also validate pair identity, order, schema and finite values.
    """
    source = Path(source)
    content_hash = sha256_file(source)
    if not _sidecar(source).exists():
        return dict(provenance="external/content-hashed", sha256=content_hash)
    from data.features import strings
    from data import reference
    metadata = json.loads(_sidecar(source).read_text())
    expected = _base_metadata(dataset, side, source_k, "strings")
    expected["code_sha256"] = _code_hash(strings, reference)
    expected["raw_sha256"] = _sources(
        [Path(config.BENCH_DIR) / dataset / (table + ".csv") for table in ("left", "right")],
        Path(config.BENCH_DIR))
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("String feature sources or implementation changed: " + str(source))
    if metadata.get("sha256") != content_hash:
        raise ValueError("String feature cache was modified: " + str(source))
    return dict(metadata, provenance="local/generated")


def _build_strings(dataset, side, k, ids, li, ri, n_left, raw_hashes, n_jobs):
    from data.features import strings
    from data import reference
    path = Path(config.FEATURES_DIR) / (config.cell_name(dataset, side, k) + ".csv.gz")
    metadata = dict(_base_metadata(dataset, side, k, "strings"), raw_sha256=raw_hashes,
                    pairs_sha256=hashlib.sha256(ids.to_csv(index=False).encode()).hexdigest(),
                    code_sha256=_code_hash(strings, reference))
    if _reusable(path, metadata):
        return
    left, right = reference.load_reference_tables(dataset, config.BENCH_DIR)
    # Conversion BEFORE concatenation preserves left int years vs right floats.
    entities = pd.concat([left.astype(object), right.astype(object)], ignore_index=True)
    attributes = [a for a in left.columns if a != "id" and a in right.columns]
    features = strings.build_zeroer_features(np.column_stack([li, n_left + ri]), entities,
        attributes, dataset_limit=n_left, drop_zero_variance=False, n_jobs=n_jobs)
    if len(features) != len(ids) or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("String extraction returned invalid values")
    _write_frame(pd.concat([ids, features], axis=1), path, metadata)


def _build_bank(dataset, side, k, ids, li, ri, models, raw_hashes, chunk_size=2048):
    from data.features import embeddings
    directory = Path(config.BANK_DIR) / config.cell_name(dataset, side, k)
    directory.mkdir(parents=True, exist_ok=True)
    pair_hash = hashlib.sha256(ids.to_csv(index=False).encode()).hexdigest()
    manifest_path = directory / "manifest.json"
    manifest = dict(_base_metadata(dataset, side, k, "bank"), pairs_sha256=pair_hash,
                    n_pairs=len(ids), attributes=DATASETS[dataset]["attributes"],
                    metrics=["cos", "linf"], models={})
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(old.get(key) != value for key, value in manifest.items() if key != "models"):
            raise ValueError("Bank candidates changed; use a fresh --output-dir")
        manifest = old
    table_sizes = [len(v) for v in _tables(dataset)]
    for model in models:
        path = directory / (model + ".parquet")
        source_paths = [Path(config.ATTR_DIR) / dataset / (model + "_" + attribute + "_" + table + ".npy")
                        for attribute in DATASETS[dataset]["attributes"] for table in ("left", "right")]
        metadata = dict(_base_metadata(dataset, side, k, "similarities"), model=model,
                        pairs_sha256=pair_hash, raw_sha256=raw_hashes,
                        sources_sha256=_sources(source_paths, Path(config.BENCH_DIR)),
                        code_sha256=_code_hash(embeddings))
        pairs_path = directory / "pairs.parquet"
        pairs_metadata = dict(_base_metadata(dataset, side, k, "bank_pairs"),
                              pairs_sha256=pair_hash)
        if _reusable(path, metadata) and _reusable(pairs_path, pairs_metadata):
            manifest["models"][model] = dict(sha256=sha256_file(path),
                                            sources_sha256=metadata["sources_sha256"], dtype="float32")
            _json(manifest_path, manifest)
            continue
        columns, presence = {}, {}
        for attribute in DATASETS[dataset]["attributes"]:
            vectors = [np.load(Path(config.ATTR_DIR) / dataset /
                (model + "_" + attribute + "_" + table + ".npy"), mmap_mode="r", allow_pickle=False)
                for table in ("left", "right")]
            if (any(len(vectors[i]) != table_sizes[i] for i in (0, 1)) or
                    vectors[0].shape[1:] != vectors[1].shape[1:]):
                raise ValueError("Attribute vectors do not match the raw tables: " + attribute)
            there = np.zeros(len(ids), dtype=bool)
            scores = {metric: np.empty(len(ids), dtype=np.float32) for metric in ("cos", "linf")}
            for start in range(0, len(ids), chunk_size):
                stop = min(start + chunk_size, len(ids))
                u = np.asarray(vectors[0][li[start:stop]], dtype=np.float32)
                v = np.asarray(vectors[1][ri[start:stop]], dtype=np.float32)
                if not np.isfinite(u).all() or not np.isfinite(v).all():
                    raise ValueError("Attribute vectors contain nonfinite values")
                both = u.any(axis=1) & v.any(axis=1)
                there[start:stop] = both
                for metric in scores:
                    scores[metric][start:stop] = np.where(both, embeddings.METRICS[metric](u, v), 0.)
            presence["present_" + attribute] = there
            for metric, values in scores.items():
                columns["{}_emb_{}__{}".format(attribute, metric, model)] = values
        pairs = ids.assign(rank=np.arange(len(ids)) % k, **presence)
        if pairs_path.exists():
            _reusable(pairs_path, pairs_metadata)
            pd.testing.assert_frame_equal(pd.read_parquet(pairs_path), pairs)
        else:
            _write_frame(pairs, pairs_path, pairs_metadata)
        _write_frame(pd.DataFrame(columns), path, metadata)
        manifest["models"][model] = dict(sha256=sha256_file(path), sources_sha256=metadata["sources_sha256"],
                                        dtype="float32")
        _json(manifest_path, manifest)
        print("Prepared similarities: {} {}".format(config.cell_name(dataset, side, k), model), flush=True)


def prepare_inputs(scope="all", datasets=None, models=None, n_jobs=1):
    """Materialize string features and embedding banks for frozen candidate cells."""
    from exp import blocking
    cells = select_cells(scope, datasets)
    models = list(MODELS if models is None else models)
    if not models or len(set(models)) != len(models) or set(models) - set(MODELS):
        raise ValueError("models must select distinct frozen model stems")
    sources = list(dict.fromkeys((d, s, _pair_source(d, s, k)) for d, s, k in cells))
    for dataset, side, k in sources:
        tables = _tables(dataset)
        raw_paths = [Path(config.BENCH_DIR) / dataset / (table + ".csv") for table in ("left", "right")]
        raw_hashes = _sources(raw_paths, Path(config.BENCH_DIR))
        vector_paths = [Path(config.RECORD_DIR) / dataset / (config.MODEL + "_" + table + ".npy")
                        for table in ("left", "right")]
        metadata = dict(_base_metadata(dataset, side, k, "pairs"), raw_sha256=raw_hashes,
                        sources_sha256=_sources(vector_paths, Path(config.BENCH_DIR)),
                        code_sha256=_code_hash(blocking), model=config.MODEL, distance=config.DISTANCE)
        path = Path(config.PAIRS_DIR) / (config.cell_name(dataset, side, k) + ".csv")
        if _reusable(path, metadata):
            ids = pd.read_csv(path, dtype=str)
            positions = [pd.Series(np.arange(len(t)), index=t) for t in tables]
            li, ri = [positions[i].loc[ids[IDS[i]]].to_numpy() for i in (0, 1)]
        else:
            ids, li, ri = _retrieve(dataset, side, k, tables)
            _write_frame(ids, path, metadata)
        for _, _, requested in [cell for cell in cells if cell[:2] == (dataset, side)
                               and _pair_source(*cell) == k]:
            if requested == k:
                continue
            selected = ids.loc[np.arange(len(ids)) % k < requested].reset_index(drop=True)
            target = Path(config.PAIRS_DIR) / (config.cell_name(dataset, side, requested) + ".csv")
            child_metadata = dict(metadata, k=requested, source_k=k)
            if not _reusable(target, child_metadata):
                _write_frame(selected, target, child_metadata)
        _build_strings(dataset, side, k, ids, li, ri, len(tables[0]), raw_hashes, n_jobs)
        _build_bank(dataset, side, k, ids, li, ri, models, raw_hashes)
        print("Prepared {}".format(config.cell_name(dataset, side, k)), flush=True)
    return [config.cell_name(*cell) for cell in cells]
