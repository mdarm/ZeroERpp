"""Regenerate the historical text-encoding recipes in a fresh benchmark tree.

Text preparation preserves pandas CSV inference, field order, prefixes, sorted
attribute deduplication and exact zero rows for missing attributes. The two
inference adapters consume texts and return normalized [text, feature] arrays
in the backend's output dtype (float32 for the heavy adapter). No fitting or
labels are involved. The CLI assembles these operations;
model computation does not acquire tables or manage experiment outputs.

Original code cells and hardware metadata are in scripts/encoding/source/.
Historical checkpoints and package versions were not pinned: these recipes
cannot guarantee bitwise regeneration of the released publication vectors.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd

from data.benchmarks import DATASETS
from exp.protocol import MODELS

PLM_MODELS = {
    "minilm6": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "minilm12": ("sentence-transformers/all-MiniLM-L12-v2", ""),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", ""),
    "distilroberta": ("sentence-transformers/all-distilroberta-v1", ""),
    "st5_xl": ("sentence-transformers/sentence-t5-xl", ""),
    "gtr_t5_large": ("sentence-transformers/gtr-t5-large", ""),
    "e5_large_v2": ("intfloat/e5-large-v2", "query: "),
    "e5_instruct": ("intfloat/multilingual-e5-large-instruct",
                    "Instruct: Retrieve semantically similar text.\nQuery: "),
    "bge_large": ("BAAI/bge-large-en-v1.5",
                  "Represent this sentence for searching relevant passages: "),
    "gte_large": ("thenlper/gte-large", ""),
}
LLM_MODELS = {stem: stem.replace("_", "/", 1) for stem in MODELS if stem not in PLM_MODELS}


def dataframe_to_text(df, attributes):
    """Serialize records in CSV row order; IDs and unlisted fields are excluded."""
    texts = []
    for row in df[attributes].itertuples(index=False):
        parts = []
        for attribute, value in zip(attributes, row):
            if pd.isna(value):
                continue
            value = str(value).strip()
            if value:
                parts.append(f"{attribute}: {value}")
        texts.append(" | ".join(parts))
    return texts


def attribute_column(df, attribute):
    """Sorted unique texts, inverse positions for present rows, and presence."""
    values = df[attribute].astype("string").str.strip()
    present = (df[attribute].notna() & values.notna() & (values != "")).to_numpy(dtype=bool)
    texts = (attribute + ": " + values[present]).to_numpy()
    unique, inverse = np.unique(texts, return_inverse=True)
    return unique.tolist(), inverse, present


def scatter(vectors, inverse, present, n_rows):
    """Restore CSV row order; missing attribute vectors remain exactly zero."""
    full = np.zeros((n_rows, vectors.shape[1]), dtype=vectors.dtype)
    full[present] = vectors[inverse]
    return full


def last_token_pooling(hidden, attention_mask):
    """Original pooling supports both left and right tokenizer padding."""
    import torch
    if attention_mask[:, -1].all():
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]


@contextmanager
def build_encoder(stem, checkpoint=None, revision=None, devices=("cuda:0", "cuda:1")):
    """Build text -> unit-vector inference; release GPU state after this model."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Fresh encoding needs optional dependencies: "
                           "python -m pip install -r requirements-encoding.txt") from exc
    common = dict(revision=revision) if revision else {}
    model = None
    try:
        if stem in PLM_MODELS:
            from sentence_transformers import SentenceTransformer
            repository, prefix = PLM_MODELS[stem]
            model = SentenceTransformer(checkpoint or repository, **common)
            dimension = model.get_sentence_embedding_dimension()

            def encode(texts, batch_size):
                if not texts:
                    return np.empty((0, dimension), dtype=np.float32)
                pool = model.start_multi_process_pool(target_devices=list(devices))
                try:
                    return model.encode([prefix + text for text in texts], pool=pool,
                        batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
                finally:
                    model.stop_multi_process_pool(pool)

            model_config = model[0].auto_model.config
            settings = dict(prefix=prefix, max_length=model.max_seq_length,
                            pooling="checkpoint-native", devices=list(devices))
        else:
            from transformers import AutoModel, AutoTokenizer
            repository = LLM_MODELS[stem]
            tokenizer = AutoTokenizer.from_pretrained(checkpoint or repository,
                trust_remote_code=True, **common)
            model = AutoModel.from_pretrained(checkpoint or repository,
                trust_remote_code=True, dtype=torch.bfloat16, device_map="auto", **common)
            model.eval()
            model_config = model.config
            dimension = getattr(model_config, "hidden_size", None)
            if dimension is None:
                dimension = model_config.text_config.hidden_size

            @torch.inference_mode()
            def encode(texts, batch_size):
                if not texts:
                    return np.empty((0, dimension), dtype=np.float32)
                batches = []
                for start in range(0, len(texts), batch_size):
                    inputs = tokenizer(texts[start:start + batch_size], padding=True,
                        truncation=True, max_length=512, return_tensors="pt")
                    inputs = {key: value.to(model.device) for key, value in inputs.items()}
                    hidden = model(**inputs).last_hidden_state
                    pooled = last_token_pooling(hidden, inputs["attention_mask"])
                    batches.append(torch.nn.functional.normalize(pooled.float(), p=2, dim=1).cpu().numpy())
                return np.concatenate(batches, axis=0)

            settings = dict(prefix="", max_length=512, pooling="last-token",
                dtype="bfloat16", padding_side=tokenizer.padding_side, device_map="auto")
        info = dict(repository=repository, checkpoint=checkpoint or repository,
                    requested_revision=revision,
                    resolved_revision=getattr(model_config, "_commit_hash", None),
                    dimension=dimension, **settings)
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        yield encode, info
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def encode_benchmarks(data_dir, output_dir, models=None, datasets=None, phases=("record", "attribute"),
                      checkpoints=None, revisions=None, devices=("cuda:0", "cuda:1"),
                      encoder_factory=build_encoder):
    """Write a new benchmark tree, preserving source tables and row alignment.

    Output must not exist. Checkpoints/revisions map frozen filename stems to
    local model directories or immutable Hub revisions. The factory seam lets
    preparation and persistence be tested without model downloads or labels.
    """
    models = list(MODELS if models is None else models)
    datasets = list(DATASETS if datasets is None else datasets)
    for selected, allowed, label in ((models, MODELS, "models"), (datasets, DATASETS, "datasets"),
                                    (list(phases), ("record", "attribute"), "phases")):
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(allowed):
            raise ValueError("Select distinct known " + label)
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    sources = [data_dir / dataset / (side + ".csv") for dataset in datasets for side in ("left", "right")]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=False)
    packages = {}
    for package in ("numpy", "pandas", "torch", "transformers", "sentence-transformers", "accelerate"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    manifest = dict(schema="zeroerpp-encoding-v1", status="in_progress",
        requested=dict(models=models, datasets=datasets, phases=list(phases)),
        started=datetime.now(timezone.utc).isoformat(),
        historical_checkpoint_revisions="not recorded in original notebooks",
        python=platform.python_version(), platform=platform.platform(), packages=packages,
        code_sha256=sha256(__file__), models={}, sources={}, outputs={})
    for dataset in datasets:
        directory = output_dir / dataset
        directory.mkdir()
        for filename in ("left.csv", "right.csv", "matches.csv"):
            source = data_dir / dataset / filename
            if source.exists():
                shutil.copyfile(source, directory / filename)
                manifest["sources"][str(source.relative_to(data_dir))] = sha256(source)
        if "attribute" in phases:
            masks = {}
            for side in ("left", "right"):
                frame = pd.read_csv(directory / (side + ".csv")).rename(
                    columns=DATASETS[dataset].get("rename_1" if side == "left" else "rename_2", {}))
                masks.update({attribute + "_" + side: attribute_column(frame, attribute)[2]
                              for attribute in DATASETS[dataset]["attributes"]})
            presence_path = output_dir / "embeddings" / "attribute" / dataset / "presence.npz"
            presence_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(presence_path, **masks)
            manifest["outputs"][str(presence_path.relative_to(output_dir))] = dict(
                kind="presence", sha256=sha256(presence_path))
    manifest_path = output_dir / "encoding_manifest.json"
    _write_json(manifest_path, manifest)
    for stem in models:
        with encoder_factory(stem, checkpoint=(checkpoints or {}).get(stem),
                revision=(revisions or {}).get(stem), devices=devices) as (encode, info):
            manifest["models"][stem] = info
            for dataset in datasets:
                attributes = DATASETS[dataset]["attributes"]
                for side in ("left", "right"):
                    frame = pd.read_csv(output_dir / dataset / (side + ".csv"))
                    frame = frame.rename(columns=DATASETS[dataset].get("rename_1" if side == "left" else "rename_2", {}))
                    for phase in phases:
                        directory = output_dir / "embeddings" / phase / dataset
                        directory.mkdir(parents=True, exist_ok=True)
                        for attribute in ([None] if phase == "record" else attributes):
                            if attribute is None:
                                texts, inverse, present = dataframe_to_text(frame, attributes), None, None
                            else:
                                texts, inverse, present = attribute_column(frame, attribute)
                            batch_size = (128 if phase == "record" else 256) if stem in PLM_MODELS else (32 if phase == "record" else 64)
                            started = time.perf_counter()
                            vectors = np.asarray(encode(texts, batch_size))
                            seconds = time.perf_counter() - started
                            if (vectors.shape != (len(texts), info["dimension"])
                                    or not np.issubdtype(vectors.dtype, np.floating)
                                    or not np.isfinite(vectors).all()
                                    or (len(texts) and not np.allclose(
                                        np.linalg.norm(vectors.astype(np.float32), axis=1), 1.,
                                        atol=max(1e-4, np.finfo(vectors.dtype).eps)))):
                                raise ValueError("Encoder must return finite normalized [text, feature] vectors")
                            full = vectors if attribute is None else scatter(vectors, inverse, present, len(frame))
                            name = stem + ("_" + attribute if attribute else "") + "_" + side + ".npy"
                            path = directory / name
                            np.save(path, full, allow_pickle=False)
                            manifest["outputs"][str(path.relative_to(output_dir))] = dict(
                                sha256=sha256(path), shape=list(full.shape), dtype=str(full.dtype), encoded_texts=len(texts),
                                batch_size=batch_size, encode_seconds=seconds)
                            _write_json(manifest_path, manifest)
                            print("Encoded " + str(path), flush=True)
    manifest["completed"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "completed_requested_scope"
    _write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Root containing dataset/left.csv and right.csv")
    parser.add_argument("--output-dir", help="Fresh benchmark root; must not exist")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--phases", nargs="+", choices=["record", "attribute"], default=["record", "attribute"])
    parser.add_argument("--checkpoints", type=Path, help="JSON mapping model stems to local checkpoint paths or Hub IDs")
    parser.add_argument("--revisions", type=Path, help="JSON mapping model stems to immutable Hub revisions")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"],
                        help="SentenceTransformer pool devices; heavy models retain device_map=auto")
    args = parser.parse_args()
    output = args.output_dir or str(Path("results") / ("encoded-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    encode_benchmarks(args.data_dir, output, args.models, args.datasets, args.phases,
        json.loads(args.checkpoints.read_text()) if args.checkpoints else None,
        json.loads(args.revisions.read_text()) if args.revisions else None, args.devices)


if __name__ == "__main__":
    main()
