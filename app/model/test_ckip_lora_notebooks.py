import ast
import json
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
        "SEED": 42,
        "TEXT_COLUMN": "data",
    }
    exec(compile(ast.Module(selected, type_ignores=[]), "<training-data>", "exec"), namespace)
    return namespace


class CKIPLoraNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_inference_contract()
        cls.training_data = load_training_data_contract()

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

    def test_synthetic_pairs_match_expected_real_sources(self):
        data_dir = MODEL_DIR.parent / "data" / "ori_data"
        real = pd.concat(
            [
                pd.read_csv(data_dir / "vpesg4k_train_1000 V1.csv"),
                pd.read_csv(data_dir / "vpesg4k_val_1000.csv"),
            ],
            ignore_index=True,
        )
        synthetic = pd.read_csv(data_dir / "augmented_misleading_data.csv")
        paired = self.training_data["attach_source_pairs"](synthetic, real)
        self.assertEqual(int(paired["pair_valid"].sum()), 105)

    def test_notebooks_are_generated_v3_and_compile(self):
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
        self.assertIn('"artifact_version\\": 3', train_source)
        self.assertIn("use_rslora=True", train_source)
        self.assertIn("assert_save_load_parity", train_source)
        self.assertIn("FULL_DATA_SEEDS = [42, 123, 2026]", train_source)


if __name__ == "__main__":
    unittest.main()
