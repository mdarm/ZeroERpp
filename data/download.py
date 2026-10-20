"""Download pinned public inputs and check their publication SHA-256 manifest.

Raw tables come from an immutable public Git revision. Vectors come from Kaggle
dataset version 3, with the archive's nested paths normalized to our data tree.
No feature matrices or candidate sets are downloaded.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import zipfile

from exp import config
from exp.protocol import MODELS, select_cells

MANIFEST = Path(__file__).with_name("input-manifest.json")


def _manifest():
    return json.loads(MANIFEST.read_text())


def _selected(manifest, datasets=None, models=None, scope="all", kind="all"):
    if kind not in ("raw", "records", "attributes", "all"):
        raise ValueError("kind must be raw, records, attributes or all")
    selected_datasets = {cell[0] for cell in select_cells(scope, datasets)}
    selected_models = set(MODELS if models is None else models)
    if not selected_models or selected_models - set(MODELS):
        raise ValueError("models must select frozen model stems")
    records = selected_models | {config.MODEL}
    return [item for item in manifest["files"]
            if item["dataset"] in selected_datasets
            and (kind == "all" or item["kind"] == kind)
            and (item["kind"] == "raw" or
                 item["model"] in (records if item["kind"] == "records" else selected_models))]


def _download_url(manifest, item):
    if item["kind"] == "raw":
        source = manifest["raw_source"]
        return "https://raw.githubusercontent.com/{}/{}/{}/{}".format(
            source["repository"], source["commit"], source["path"], item["path"])
    source = manifest["embeddings_source"]
    # Kaggle needs an encoded filename segment: literal internal slashes 404.
    return "https://www.kaggle.com/api/v1/datasets/download/{}/{}?{}".format(
        source["handle"], quote(item["kaggle_path"], safe=""),
        urlencode({"datasetVersionNumber": source["version"]}))


def _check_file(path, item):
    from data.prepare import sha256_file
    if not path.is_file():
        raise FileNotFoundError(str(path) + "; run python main_zeroer.py download first")
    if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
        raise ValueError("Input checksum mismatch: " + str(path))


def _fetch(url, path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".download")
    request = Request(url, headers={"User-Agent": "ZeroERpp-publication-replication/1.0"})
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    except (HTTPError, URLError) as error:
        raise RuntimeError("Public download failed for {} ({}). Sources, file paths and checksums "
                           "are listed in data/input-manifest.json."
                           .format(item["path"], error)) from error
    # Kaggle may return an individual payload or a single-file zip.
    if zipfile.is_zipfile(temporary):
        extracted = Path(str(path) + ".part")
        with zipfile.ZipFile(temporary) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if len(members) != 1 or Path(members[0]).name != path.name:
                raise ValueError("Unexpected download archive members for " + str(path))
            with archive.open(members[0]) as source, extracted.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        temporary.unlink()
        temporary = extracted
    _check_file(temporary, item)
    temporary.replace(path)


def download_data(datasets=None, models=None, scope="all", kind="all", data_root=None):
    """Download only requested inputs; complete existing inputs are hash-checked."""
    manifest = _manifest()
    root = Path(data_root or config.BENCH_DIR)
    items = _selected(manifest, datasets, models, scope, kind)
    pending = sum(item["bytes"] for item in items if not (root / item["path"]).exists())
    print("Checking {} input files; {:.2f} GB to download".format(len(items), pending / 1e9), flush=True)
    for item in items:
        path = root / item["path"]
        if path.exists():
            _check_file(path, item)
            continue
        print("Downloading " + item["path"], flush=True)
        _fetch(_download_url(manifest, item), path, item)
    return {"files": len(items), "data_root": str(root), "verified": True}


def verify_data(datasets=None, models=None, scope="all", kind="all", data_root=None,
                verify_embeddings=True):
    """Verify hashes and vector/table alignment; evaluation labels are checked only here."""
    import numpy as np
    import pandas as pd
    manifest = _manifest()
    root = Path(data_root or config.BENCH_DIR)
    items = _selected(manifest, datasets, models, scope, kind if verify_embeddings else "raw")
    table_rows = {}
    for item in items:
        path = root / item["path"]
        _check_file(path, item)
        if item["kind"] == "raw":
            frame = pd.read_csv(path, dtype=str)
            if path.name in ("left.csv", "right.csv"):
                if frame["id"].isna().any() or frame["id"].duplicated().any():
                    raise ValueError("Table IDs must be nonmissing and unique: " + str(path))
                table_rows[item["dataset"], path.stem] = len(frame)
            continue
        vectors = np.load(path, mmap_mode="r", allow_pickle=False)
        side = path.stem.rsplit("_", 1)[1]
        key = item["dataset"], side
        if key not in table_rows:
            table_rows[key] = len(pd.read_csv(root / item["dataset"] / (side + ".csv"), usecols=["id"]))
        if (list(vectors.shape) != item["shape"] or str(vectors.dtype) != item["dtype"]
                or len(vectors) != table_rows[key]):
            raise ValueError("Embedding shape, dtype or table row count mismatch: " + str(path))
    print("Verified {} public input files".format(len(items)), flush=True)
    return {"files": len(items), "data_root": str(root), "verified": True}
