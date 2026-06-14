import ast
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.model import build_ckip_lora_notebooks as builder


MODEL_DIR = Path(__file__).resolve().parent
DATA_DIR = MODEL_DIR.parent / "data"


def selected_namespace(source, functions, assignments, namespace):
    tree = ast.parse(source)
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if names & assignments:
                selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name in functions:
                selected.append(node)
    exec(
        compile(ast.Module(selected, type_ignores=[]), "<contract>", "exec"),
        namespace,
    )
    return namespace


def load_data_contract():
    namespace = {
        "pd": pd,
        "Path": Path,
        "LOCAL_DATA_DIR": DATA_DIR,
        "RAW_BASE_URL": "unused://",
        "ID_COLUMN": "id",
        "TEXT_COLUMN": "data",
        "TARGET_COLUMNS": [
            "promise_status",
            "verification_timeline",
            "evidence_status",
            "evidence_quality",
        ],
        "MLM_CORPUS_FILES": {
            "train": (
                Path("ori_data") / "vpesg4k_train_1000 V1.csv",
                1000,
            ),
            "val": (
                Path("ori_data") / "vpesg4k_val_1000.csv",
                1000,
            ),
            "test": (
                Path("ori_data") / "vpesg4k_test_2000.csv",
                2000,
            ),
        },
        "FOLDS": [1, 2, 3, 4, 5],
    }
    return selected_namespace(
        builder.TRAIN_DATA,
        functions={
            "normalize_value",
            "read_csv_local_or_remote",
            "load_mlm_corpus",
            "load_fold_data",
        },
        assignments=set(),
        namespace=namespace,
    )


def load_mlm_contract():
    namespace = {
        "TEXT_COLUMN": "data",
        "MLM_MAX_LEN": 8,
        "MLM_STRIDE": 2,
        "tqdm": lambda iterable, desc=None: iterable,
    }
    return selected_namespace(
        builder.TRAIN_MLM,
        functions={"build_mlm_examples"},
        assignments=set(),
        namespace=namespace,
    )


def load_inference_contract():
    namespace = {
        "np": np,
        "pd": pd,
    }
    return selected_namespace(
        builder.INFERENCE_MAIN,
        functions={"route_predictions", "validate_input"},
        assignments={
            "ID_COLUMN",
            "TEXT_COLUMN",
            "TARGET_COLUMNS",
            "TASK_CLASSES",
        },
        namespace=namespace,
    )


class FakeOverflowTokenizer:
    def __call__(
        self,
        text,
        add_special_tokens,
        truncation,
        max_length,
        stride,
        return_overflowing_tokens,
        return_attention_mask,
        return_special_tokens_mask,
    ):
        del (
            add_special_tokens,
            truncation,
            return_overflowing_tokens,
            return_attention_mask,
            return_special_tokens_mask,
        )
        body = list(range(10, 10 + len(text)))
        capacity = max_length - 2
        chunks = []
        start = 0
        while start < len(body):
            chunk = body[start : start + capacity]
            ids = [101] + chunk + [102]
            chunks.append(ids)
            if start + capacity >= len(body):
                break
            start += capacity - stride
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * len(ids) for ids in chunks],
            "special_tokens_mask": [
                [1] + [0] * (len(ids) - 2) + [1]
                for ids in chunks
            ],
        }


class SimplifiedCKIPNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data_contract()
        cls.mlm = load_mlm_contract()
        cls.inference = load_inference_contract()

    def test_mlm_corpus_is_exact_official_4000(self):
        corpus = self.data["load_mlm_corpus"]()
        self.assertEqual(list(corpus.columns), ["data"])
        self.assertEqual(len(corpus), 4000)
        self.assertFalse(corpus["data"].isna().any())
        self.assertTrue(corpus["data"].astype(str).str.strip().ne("").all())
        paths = {
            path.name
            for path, _ in self.data["MLM_CORPUS_FILES"].values()
        }
        self.assertNotIn("augmented_misleading_data.csv", paths)

    def test_classification_uses_five_prepared_folds(self):
        validation_ids = set()
        for fold in range(1, 6):
            train_df, val_df = self.data["load_fold_data"](fold)
            self.assertEqual(len(train_df), 1711)
            self.assertEqual(len(val_df), 400)
            synthetic = pd.to_numeric(train_df["id"]) >= 90000
            self.assertEqual(int(synthetic.sum()), 111)
            self.assertFalse(
                pd.to_numeric(val_df["id"]).ge(90000).any()
            )
            self.assertFalse(validation_ids & set(val_df["id"]))
            validation_ids.update(val_df["id"])
        self.assertEqual(len(validation_ids), 2000)

    def test_mlm_overflow_keeps_overlap(self):
        corpus = pd.DataFrame({"data": ["abcdefghijkl"]})
        examples = self.mlm["build_mlm_examples"](
            corpus,
            FakeOverflowTokenizer(),
        )
        self.assertEqual(len(examples), 3)
        first_body = examples[0]["input_ids"][1:-1]
        second_body = examples[1]["input_ids"][1:-1]
        self.assertEqual(first_body[-2:], second_body[:2])

    def test_fixed_threshold_and_argmax_routing(self):
        test_df = pd.DataFrame(
            {"id": [1, 2, 3], "data": ["a", "b", "c"]}
        )
        probabilities = {
            "t1": np.array(
                [[0.49, 0.51], [0.50, 0.50], [0.90, 0.10]]
            ),
            "t2": np.array(
                [
                    [0.1, 0.7, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.7],
                    [0.7, 0.1, 0.1, 0.1],
                ]
            ),
            "t3": np.array(
                [[0.49, 0.51], [0.80, 0.20], [0.10, 0.90]]
            ),
            "t4": np.array(
                [
                    [0.1, 0.2, 0.7],
                    [0.7, 0.2, 0.1],
                    [0.7, 0.2, 0.1],
                ]
            ),
        }
        output = self.inference["route_predictions"](
            test_df,
            probabilities,
            t1_threshold=0.5,
            t3_threshold=0.5,
        )
        self.assertEqual(output.loc[0, "promise_status"], "Yes")
        self.assertEqual(
            output.loc[0, "verification_timeline"],
            "within_2_years",
        )
        self.assertEqual(output.loc[0, "evidence_quality"], "Misleading")
        self.assertEqual(output.loc[1, "evidence_status"], "No")
        self.assertEqual(output.loc[1, "evidence_quality"], "N/A")
        self.assertEqual(output.loc[2, "promise_status"], "No")

    def test_notebooks_compile_and_use_simple_artifact(self):
        for name in ["model_train.ipynb", "model_inference.ipynb"]:
            notebook = json.loads(
                (MODEL_DIR / name).read_text(encoding="utf-8")
            )
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    compile(
                        "".join(cell["source"]),
                        f"{name}:cell{index}",
                        "exec",
                    )
        train_source = "\n".join(
            "".join(cell["source"])
            for cell in builder.build_train_notebook()["cells"]
        )
        inference_source = "\n".join(
            "".join(cell["source"])
            for cell in builder.build_inference_notebook()["cells"]
        )
        self.assertIn('"artifact_version": 8', train_source)
        self.assertIn("AutoModelForMaskedLM", train_source)
        self.assertIn("get_peft_model", train_source)
        self.assertIn("self.shared_mlp", train_source)
        self.assertIn("self.heads", train_source)
        self.assertIn("self.shared_mlp(hidden[:, 0, :])", train_source)
        self.assertIn("F.cross_entropy", train_source)
        self.assertIn('"t1_yes": T1_THRESHOLD', train_source)
        self.assertIn('"t3_yes": T3_THRESHOLD', train_source)
        self.assertIn('"multiclass_decision": "argmax"', train_source)
        self.assertIn("compute_class_weights", train_source)
        self.assertIn("weight=class_weights[task]", train_source)
        self.assertIn('"ensemble_members": [', train_source)
        self.assertIn("ensemble_probabilities", inference_source)
        self.assertIn("only supports simplified artifact v8", inference_source)

    def test_long_training_uses_differential_learning_rates(self):
        source = builder.TRAIN_CONFIG + builder.TRAIN_CLASSIFIER
        self.assertIn("MLM_EPOCHS = 6", source)
        self.assertIn("MLM_LR = 3e-5", source)
        self.assertIn("MAX_EPOCHS = 30", source)
        self.assertIn("EARLY_STOPPING_PATIENCE = 5", source)
        self.assertIn("LORA_LEARNING_RATE = 5e-5", source)
        self.assertIn('"lr": LORA_LEARNING_RATE', source)
        self.assertIn('"lr": HEAD_LEARNING_RATE', source)
        self.assertIn('"lora_learning_rate": current_lrs[0]', source)
        self.assertIn('"head_learning_rate": current_lrs[1]', source)

    def test_cls_head_matches_inference(self):
        train_source = builder.TRAIN_CLASSIFIER
        inference_source = builder.INFERENCE_MAIN
        expected = [
            "nn.Linear(hidden_size, 256)",
            "nn.Linear(256, 128)",
            "self.shared_mlp(hidden[:, 0, :])",
        ]
        for token in expected:
            self.assertIn(token, train_source)
            self.assertIn(token, inference_source)

    def test_checkpoint_uses_competition_score_weights(self):
        source = builder.TRAIN_CONFIG + builder.TRAIN_CLASSIFIER
        self.assertIn('"promise_status": 0.20', source)
        self.assertIn('"verification_timeline": 0.15', source)
        self.assertIn('"evidence_status": 0.30', source)
        self.assertIn('"evidence_quality": 0.35', source)
        self.assertIn('scores["competition_macro_f1"]', source)
        self.assertIn(
            'metrics["competition_macro_f1"] > best_score',
            source,
        )

    def test_complex_training_features_are_removed(self):
        source = "\n".join(
            [
                builder.TRAIN_CONFIG,
                builder.TRAIN_DATA,
                builder.TRAIN_MLM,
                builder.TRAIN_CLASSIFIER,
                builder.TRAIN_RUN,
                builder.TRAIN_ARTIFACT,
            ]
        )
        forbidden = [
            "FULL_DATA_SEEDS",
            "fit_scalar_temperatures",
            "tune_routing_thresholds",
            "tune_t2_class_biases",
            "tune_misleading_thresholds",
            "FOCAL_GAMMA",
            "ordinal",
            "pair_loss",
            "multi_sample",
            "synthetic_df",
        ]
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_mlm_runs_before_five_fold_classifiers(self):
        source = builder.TRAIN_RUN
        self.assertLess(
            source.index(
                "mlm_history, mlm_config = "
                "train_mlm_backbone(mlm_corpus"
            ),
            source.index(
                "model, fold_history, metrics, predictions = "
                "train_classifier("
            ),
        )
        self.assertIn("for fold in FOLDS:", source)


if __name__ == "__main__":
    unittest.main()
