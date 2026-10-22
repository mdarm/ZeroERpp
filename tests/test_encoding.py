"""Verify historical preprocessing and the inference/output boundary offline."""
import ast
from contextlib import contextmanager
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data import encoding
from data.benchmarks import DATASETS
from exp.protocol import MODELS


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "encoding" / "source"


def original_functions(path):
    notebook = json.loads(path.read_text())
    source = "\n\n".join("".join(cell["source"]) for cell in notebook["cells"]
                         if cell["cell_type"] == "code")
    source = "\n".join(line for line in source.splitlines() if not line.startswith("!"))
    names = {"dataframe_to_text", "attribute_column", "scatter"}
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(np=np, pd=pd)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class EncodingTests(unittest.TestCase):
    def test_preparation_matches_both_original_notebooks(self):
        frame = pd.read_csv(io.StringIO("id,name,year,unused\n1, Alpha ,2001,leak\n2,,2001,leak\n3,Alpha,,leak\n4,   ,2002,leak\n5,Beta,2001,leak\n"))
        for path in SOURCE.glob("*.ipynb"):
            with self.subTest(path=path.name):
                original = original_functions(path)
                self.assertEqual(encoding.dataframe_to_text(frame, ["name", "year"]),
                                 original["dataframe_to_text"](frame, ["name", "year"]))
                for attribute in ("name", "year"):
                    actual = encoding.attribute_column(frame, attribute)
                    expected = original["attribute_column"](frame, attribute)
                    self.assertEqual(actual[0], expected[0])
                    np.testing.assert_array_equal(actual[1], expected[1])
                    np.testing.assert_array_equal(actual[2], expected[2])
                    vectors = np.arange(len(actual[0]) * 3).reshape(-1, 3).astype(np.float32)
                    np.testing.assert_array_equal(encoding.scatter(vectors, actual[1], actual[2], len(frame)),
                        original["scatter"](vectors, expected[1], expected[2], len(frame)))

    def test_recipes_cover_all_historical_models_and_prefixes(self):
        self.assertEqual(set(MODELS), set(encoding.PLM_MODELS) | set(encoding.LLM_MODELS))
        notebook = json.loads((SOURCE / "small-llm-encoding.ipynb").read_text())
        source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        wanted = {"E5_INSTRUCT", "PLM_MODELS", "ATTRIBUTES", "RENAME_RIGHT"}
        assignments = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id in wanted for target in node.targets)]
        namespace = {}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), "source", "exec"), namespace)
        self.assertEqual(encoding.PLM_MODELS, namespace["PLM_MODELS"])
        self.assertEqual({dataset: spec["attributes"] for dataset, spec in DATASETS.items()}, namespace["ATTRIBUTES"])

    def test_last_token_pooling_handles_both_padding_directions(self):
        class Mask(np.ndarray):
            def sum(self, dim):
                return np.asarray(self).sum(axis=dim)

        class Hidden:
            device = "cpu"
            values = np.arange(2 * 4 * 3).reshape(2, 4, 3)

            def size(self, axis):
                return self.values.shape[axis]

            def __getitem__(self, key):
                return self.values[key]

        fake_torch = types.SimpleNamespace(arange=lambda n, device: np.arange(n))
        with patch.dict("sys.modules", torch=fake_torch):
            for mask, positions in (([[0, 0, 1, 1], [0, 1, 1, 1]], [3, 3]),
                                    ([[1, 1, 0, 0], [1, 1, 1, 0]], [1, 2])):
                result = encoding.last_token_pooling(Hidden(), np.asarray(mask).view(Mask))
                np.testing.assert_array_equal(result, Hidden.values[np.arange(2), positions])

    def test_sentence_adapter_retains_prefix_native_length_and_pool_lifecycle(self):
        calls = {}

        class SentenceModel:
            max_seq_length = 384

            def __init__(self, checkpoint, **kwargs):
                calls["load"] = (checkpoint, kwargs)

            def get_sentence_embedding_dimension(self):
                return 2

            def __getitem__(self, index):
                return types.SimpleNamespace(auto_model=types.SimpleNamespace(
                    config=types.SimpleNamespace(_commit_hash="resolved-test-revision")))

            def start_multi_process_pool(self, **kwargs):
                calls["pool"] = kwargs
                return "test-pool"

            def encode(self, texts, **kwargs):
                calls["encode"] = (texts, kwargs)
                return np.array([[.6, .8]], dtype=np.float32)

            def stop_multi_process_pool(self, pool):
                calls["stopped"] = pool

        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            device_count=lambda: 0, is_available=lambda: False))
        with patch.dict("sys.modules", torch=fake_torch,
                        sentence_transformers=types.SimpleNamespace(SentenceTransformer=SentenceModel)):
            with encoding.build_encoder("e5_large_v2", checkpoint="local-test",
                    revision="requested-test-revision", devices=["cpu"]) as (encode, metadata):
                self.assertEqual(encode([], 256).shape, (0, 2))
                encode(["title: value"], 256)
        self.assertEqual(calls["load"], ("local-test", {"revision": "requested-test-revision"}))
        self.assertEqual(calls["encode"][0], ["query: title: value"])
        self.assertTrue(calls["encode"][1]["normalize_embeddings"])
        self.assertEqual(calls["stopped"], "test-pool")
        self.assertEqual(metadata["max_length"], 384)
        self.assertEqual(metadata["resolved_revision"], "resolved-test-revision")

    def test_fresh_run_preserves_rows_missingness_inputs_and_provenance(self):
        calls = []

        @contextmanager
        def factory(stem, **kwargs):
            def encode(texts, batch_size):
                calls.append((texts, batch_size))
                return np.tile(np.array([[0.6, 0.8]], dtype=np.float16), (len(texts), 1))
            yield encode, dict(dimension=2, checkpoint="offline-test")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "amazon_googleproducts"
            source.mkdir(parents=True)
            pd.DataFrame(dict(id=["1", "2"], title=["same", "same"], description=["", ""],
                              manufacturer=[None, "Maker"], price=[1., 2.])).to_csv(source / "left.csv", index=False)
            pd.DataFrame(dict(id=["3", "4"], name=["same", "same"], description=["", ""],
                              manufacturer=["Maker", None], price=[1., 2.])).to_csv(source / "right.csv", index=False)
            # This deliberately is not parseable gold. Encoding only copies it.
            (source / "matches.csv").write_text("unused arbitrary bytes\n")
            output = root / "fresh"
            manifest = encoding.encode_benchmarks(root / "source", output, ["minilm6"],
                ["amazon_googleproducts"], encoder_factory=factory)
            self.assertEqual(len(manifest["outputs"]), 11)
            for name, metadata in manifest["outputs"].items():
                self.assertEqual(metadata["sha256"], encoding.sha256(output / name))
                if metadata.get("kind") != "presence":
                    self.assertEqual(metadata["shape"], [2, 2])
                    self.assertEqual(metadata["dtype"], "float16")
            attribute = output / "embeddings/attribute/amazon_googleproducts"
            np.testing.assert_array_equal(np.load(attribute / "minilm6_description_left.npy"), np.zeros((2, 2)))
            vectors = np.load(attribute / "minilm6_manufacturer_left.npy")
            np.testing.assert_array_equal(vectors[0], [0., 0.])
            np.testing.assert_array_equal(vectors[1], np.array([.6, .8], dtype=np.float16))
            np.testing.assert_array_equal(np.load(attribute / "presence.npz")["manufacturer_left"], [False, True])
            self.assertIn((["title: same"], 256), calls)
            self.assertTrue(any("title: same" in text for texts, _ in calls for text in texts))
            self.assertEqual((output / "amazon_googleproducts/matches.csv").read_bytes(),
                             (source / "matches.csv").read_bytes())
            self.assertIn("completed", manifest)
            self.assertEqual(manifest["status"], "completed_requested_scope")
            self.assertEqual(manifest["requested"]["models"], ["minilm6"])
            with self.assertRaises(FileExistsError):
                encoding.encode_benchmarks(root / "source", output, ["minilm6"],
                    ["amazon_googleproducts"], encoder_factory=factory)


if __name__ == "__main__":
    unittest.main()
