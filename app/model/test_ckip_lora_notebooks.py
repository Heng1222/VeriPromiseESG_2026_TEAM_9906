import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.model import build_ckip_lora_notebooks as builder


MODEL_DIR = Path(__file__).resolve().parent


def load_inference_contract():
    tree = ast.parse(builder.INFERENCE_MAIN)
    assignments = {
        "ID_COLUMN",
        "TEXT_COLUMN",
        "TARGET_COLUMNS",
        "V3_TASK_CLASSES",
        "LEGACY_TASK_CLASSES",
        "OFFICIAL_ALLOWED_VALUES",
    }
    functions = {
        "choose_t4_label",
        "route_predictions",
        "validate_input",
        "validate_output",
    }
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
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(selected, type_ignores=[]), "<contract>", "exec"), namespace)
    return namespace


def load_training_data_contract():
    tree = ast.parse(builder.TRAIN_DATA)
    functions = {
        "normalize_value",
        "read_csv_local_or_remote",
        "load_mlm_corpus",
        "read_fold_csv",
        "normalize_training_frame",
        "validate_training_frame",
        "load_real_data",
        "source_group_key",
        "assign_synthetic_folds",
        "attach_source_pairs",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in functions
    ]
    namespace = {
        "np": np,
        "pd": pd,
        "Path": Path,
        "SEED": 42,
        "FOLDS": [1, 2, 3, 4, 5],
        "LOCAL_DATA_DIR": MODEL_DIR.parent / "data",
        "RAW_BASE_URL": "unused://",
        "ID_COLUMN": "id",
        "TEXT_COLUMN": "data",
        "TARGET_COLUMNS": [
            "promise_status",
            "verification_timeline",
            "evidence_status",
            "evidence_quality",
        ],
        "SYNTHETIC_ID_MIN": 90000,
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
    }
    exec(compile(ast.Module(selected, type_ignores=[]), "<training-data>", "exec"), namespace)
    return namespace


def load_mlm_contract():
    tree = ast.parse(builder.TRAIN_MLM)
    functions = {"build_mlm_examples"}
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in functions
    ]
    namespace = {
        "TEXT_COLUMN": "data",
        "MLM_MAX_LEN": 8,
        "MLM_STRIDE": 2,
        "tqdm": lambda iterable, desc=None: iterable,
    }
    exec(compile(ast.Module(selected, type_ignores=[]), "<mlm>", "exec"), namespace)
    return namespace


def load_inference_artifact_contract():
    tree = ast.parse(builder.INFERENCE_MAIN)
    assignments = {"DEFAULT_MODEL_NAME"}
    functions = {"resolve_peft_backbone"}
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
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    namespace = {"Path": Path}
    exec(
        compile(
            ast.Module(selected, type_ignores=[]),
            "<inference-artifact>",
            "exec",
        ),
        namespace,
    )
    return namespace


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


class CKIPLoraNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_inference_contract()
        cls.training_data = load_training_data_contract()
        cls.mlm = load_mlm_contract()
        cls.inference_artifact = load_inference_artifact_contract()

    def make_probabilities(self):
        return {
            "t1": np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]),
            "t2": np.array(
                [
                    [0.05, 0.05, 0.10, 0.80],
                    [0.80, 0.05, 0.10, 0.05],
                    [0.80, 0.05, 0.10, 0.05],
                ]
            ),
            "t3": np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1]]),
            "t4": np.array(
                [
                    [0.10, 0.15, 0.75],
                    [0.80, 0.10, 0.10],
                    [0.80, 0.10, 0.10],
                ]
            ),
        }

    def test_v3_routing_and_official_timeline_label(self):
        test_df = pd.DataFrame(
            {"id": [1, 2, 3], "data": ["a", "b", "c"]}
        )
        output = self.contract["route_predictions"](
            test_df,
            self.make_probabilities(),
            self.contract["V3_TASK_CLASSES"],
            t1_threshold=0.5,
            t3_threshold=0.5,
            misleading_config={
                "probability_threshold": 0.7,
                "margin_threshold": 0.2,
            },
        )
        self.assertEqual(output.loc[0, "verification_timeline"], "more_than_5_years")
        self.assertEqual(output.loc[0, "evidence_quality"], "Misleading")
        self.assertEqual(output.loc[1, "promise_status"], "No")
        self.assertEqual(output.loc[2, "evidence_status"], "No")
        self.contract["validate_output"](output, test_df)

    def test_legacy_timeline_alias_is_exported_as_official_value(self):
        test_df = pd.DataFrame({"id": [1], "data": ["a"]})
        probabilities = {
            task: values[:1]
            for task, values in self.make_probabilities().items()
        }
        output = self.contract["route_predictions"](
            test_df,
            probabilities,
            self.contract["LEGACY_TASK_CLASSES"],
            t1_threshold=0.5,
            t3_threshold=0.5,
        )
        self.assertEqual(output.loc[0, "verification_timeline"], "more_than_5_years")

    def test_misleading_requires_probability_and_margin(self):
        choose = self.contract["choose_t4_label"]
        labels = self.contract["V3_TASK_CLASSES"]["t4"]
        config = {
            "probability_threshold": 0.6,
            "margin_threshold": 0.2,
        }
        self.assertEqual(
            choose(np.array([0.20, 0.15, 0.65]), labels, config),
            "Misleading",
        )
        self.assertEqual(
            choose(np.array([0.35, 0.10, 0.55]), labels, config),
            "Clear",
        )

    def test_v3_class_bias_and_t4_binary_threshold(self):
        test_df = pd.DataFrame({"id": [1], "data": ["a"]})
        probabilities = {
            "t1": np.array([[0.1, 0.9]]),
            "t2": np.array([[0.45, 0.40, 0.10, 0.05]]),
            "t3": np.array([[0.1, 0.9]]),
            "t4": np.array([[0.52, 0.48, 0.00]]),
        }
        output = self.contract["route_predictions"](
            test_df,
            probabilities,
            self.contract["V3_TASK_CLASSES"],
            t1_threshold=0.5,
            t3_threshold=0.5,
            misleading_config={
                "probability_threshold": 1.0,
                "margin_threshold": 1.0,
            },
            t2_class_biases={"within_2_years": 0.2},
            t4_not_clear_threshold=0.45,
        )
        self.assertEqual(
            output.loc[0, "verification_timeline"],
            "within_2_years",
        )
        self.assertEqual(output.loc[0, "evidence_quality"], "Not Clear")

    def test_output_validation_rejects_non_official_alias(self):
        test_df = pd.DataFrame({"id": [1], "data": ["a"]})
        output = pd.DataFrame(
            [
                {
                    "id": 1,
                    "promise_status": "Yes",
                    "verification_timeline": "longer_than_5_years",
                    "evidence_status": "Yes",
                    "evidence_quality": "Clear",
                }
            ]
        )
        with self.assertRaises(ValueError):
            self.contract["validate_output"](output, test_df)

    def test_training_alias_and_synthetic_group_isolation(self):
        normalize = self.training_data["normalize_value"]
        self.assertEqual(
            normalize("longer_than_5_years"),
            "more_than_5_years",
        )
        synthetic = pd.read_csv(
            MODEL_DIR.parent / "data" / "ori_data" / "augmented_misleading_data.csv"
        )
        assigned = self.training_data["assign_synthetic_folds"](synthetic)
        group_folds = assigned.groupby(
            ["pdf_url", "page_number"],
            dropna=False,
        )["synthetic_fold"].nunique()
        self.assertTrue(group_folds.eq(1).all())
        self.assertEqual(group_folds.size, 37)
        self.assertEqual(set(assigned["synthetic_fold"]), {1, 2, 3, 4, 5})

    def test_fold_loader_reconstructs_real_and_synthetic_data(self):
        real, synthetic, folds = self.training_data["load_real_data"]()
        self.assertEqual(len(real), 2000)
        self.assertEqual(real["id"].nunique(), 2000)
        self.assertEqual(len(synthetic), 111)
        self.assertEqual(synthetic["id"].nunique(), 111)
        expected_synthetic_ids = set(synthetic["id"])
        for fold in range(1, 6):
            self.assertEqual(len(folds[fold]["train_real"]), 1600)
            self.assertEqual(len(folds[fold]["val"]), 400)
            train_synthetic_ids = (
                set(folds[fold]["train"]["id"])
                - set(folds[fold]["train_real"]["id"])
            )
            self.assertEqual(train_synthetic_ids, expected_synthetic_ids)

    def test_synthetic_pairs_match_expected_real_sources(self):
        real, synthetic, _ = self.training_data["load_real_data"]()
        paired = self.training_data["attach_source_pairs"](synthetic, real)
        self.assertEqual(int(paired["pair_valid"].sum()), 105)

    def test_mlm_corpus_uses_only_official_4000_data_rows(self):
        corpus = self.training_data["load_mlm_corpus"]()
        self.assertEqual(list(corpus.columns), ["data"])
        self.assertEqual(len(corpus), 4000)
        self.assertFalse(corpus["data"].isna().any())
        self.assertTrue(corpus["data"].astype(str).str.strip().ne("").all())
        corpus_files = self.training_data["MLM_CORPUS_FILES"]
        self.assertEqual(
            [rows for _, rows in corpus_files.values()],
            [1000, 1000, 2000],
        )
        self.assertNotIn(
            "augmented_misleading_data.csv",
            {path.name for path, _ in corpus_files.values()},
        )

    def test_mlm_overflow_chunks_keep_configured_overlap(self):
        corpus = pd.DataFrame({"data": ["abcdefghijkl"]})
        examples = self.mlm["build_mlm_examples"](
            corpus,
            FakeOverflowTokenizer(),
        )
        self.assertEqual(len(examples), 3)
        self.assertTrue(all(len(row["input_ids"]) <= 8 for row in examples))
        first_body = examples[0]["input_ids"][1:-1]
        second_body = examples[1]["input_ids"][1:-1]
        self.assertEqual(first_body[-2:], second_body[:2])

    def test_v4_backbone_resolution_requires_saved_encoder(self):
        resolve = self.inference_artifact["resolve_peft_backbone"]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backbone = root / "mlm_backbone"
            backbone.mkdir()
            (backbone / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                resolve(
                    root,
                    {
                        "artifact_version": 4,
                        "backbone_path": "mlm_backbone",
                    },
                ),
                str(backbone),
            )
            self.assertEqual(
                resolve(
                    root,
                    {
                        "artifact_version": 3,
                        "model_name": "legacy/model",
                    },
                ),
                "legacy/model",
            )

    def test_notebooks_are_generated_v4_and_compile(self):
        for name in ["model_train.ipynb", "model_inference.ipynb"]:
            notebook = json.loads((MODEL_DIR / name).read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    compile(
                        "".join(cell["source"]),
                        f"{name}:cell{index}",
                        "exec",
                    )
        train_source = json.dumps(
            json.loads((MODEL_DIR / "model_train.ipynb").read_text(encoding="utf-8"))
        )
        inference_source = json.dumps(
            json.loads(
                (MODEL_DIR / "model_inference.ipynb").read_text(
                    encoding="utf-8"
                )
            )
        )
        self.assertIn('"artifact_version\\": 4', train_source)
        self.assertIn("AutoModelForMaskedLM", train_source)
        self.assertIn("DataCollatorForLanguageModeling", train_source)
        self.assertIn("return_overflowing_tokens=True", train_source)
        self.assertIn("MLM_STRIDE = 64", train_source)
        self.assertIn("MLM_PROBABILITY = 0.15", train_source)
        self.assertIn("MLM_EPOCHS = 3", train_source)
        self.assertIn("MLM_BACKBONE_DIR", train_source)
        self.assertIn("use_rslora=True", train_source)
        self.assertIn("assert_save_load_parity", train_source)
        self.assertIn("FULL_DATA_SEEDS = [42, 123, 2026]", train_source)
        self.assertIn(
            'pair_valid_cpu = batch[\\"pair_valid\\"].bool()',
            train_source,
        )
        self.assertIn(
            'batch[\\"pair_input_ids\\"][pair_valid_cpu]',
            train_source,
        )
        self.assertIn('FOCAL_GAMMA_BY_TASK', train_source)
        self.assertIn('best_optimizer_steps', train_source)
        self.assertIn('t2_class_biases', train_source)
        self.assertIn('t4_not_clear_threshold', train_source)
        self.assertIn('report_quality_metrics', train_source)
        self.assertIn("mlm_backbone/**", inference_source)
        self.assertIn("version in {3, 4}", inference_source)
        self.assertIn("resolve_peft_backbone", inference_source)
        self.assertNotIn('run_quality_gate', train_source)
        self.assertNotIn('Quality gate failed', train_source)

    def test_mlm_executes_before_any_classification_training(self):
        notebook = builder.build_train_notebook()
        code_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        source = "\n".join(code_cells)
        mlm_run = source.index(
            "mlm_history, mlm_config = train_mlm_backbone"
        )
        fold_run = source.index(
            'for fold in FOLDS:\n'
            '    print(f"\\n{\'=\' * 64}\\nTraining fold {fold}'
        )
        full_run = source.index(
            'for seed in FULL_DATA_SEEDS:\n'
            '    print(f"\\n{\'=\' * 64}\\nTraining full-data seed {seed}'
        )
        self.assertLess(mlm_run, fold_run)
        self.assertLess(mlm_run, full_run)
        self.assertIn(
            "ESGLoraMTLModel(MLM_BACKBONE_DIR)",
            source,
        )

    def test_quality_report_never_blocks_training(self):
        tree = ast.parse(builder.TRAIN_CALIBRATE)
        report_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "report_quality_metrics"
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Raise)
                for node in ast.walk(report_function)
            )
        )


if __name__ == "__main__":
    unittest.main()
