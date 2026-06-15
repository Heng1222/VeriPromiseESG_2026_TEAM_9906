import ast
import io
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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


def load_model_components(source):
    namespace = {
        "torch": torch,
        "nn": nn,
    }
    return selected_namespace(
        source,
        functions={
            "LearnableLayerPool",
            "ResidualAdapter",
            "SharedFeatureMLP",
            "TaskHead",
        },
        assignments=set(),
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
        expected_notebooks = {
            "model_train.ipynb": builder.build_train_notebook(),
            "model_inference.ipynb": builder.build_inference_notebook(),
        }
        for name, expected_notebook in expected_notebooks.items():
            notebook = json.loads(
                (MODEL_DIR / name).read_text(encoding="utf-8")
            )
            for cell in notebook["cells"]:
                cell.pop("id", None)
            self.assertEqual(notebook, expected_notebook)
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
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
        self.assertIn('"artifact_version": 9', train_source)
        self.assertIn("AutoModelForMaskedLM", train_source)
        self.assertIn("get_peft_model", train_source)
        self.assertIn("self.layer_pooler", train_source)
        self.assertIn("self.shared_mlp", train_source)
        self.assertIn("self.heads", train_source)
        self.assertIn("output_hidden_states=True", train_source)
        self.assertIn(
            "self.layer_pooler(outputs.hidden_states)",
            train_source,
        )
        self.assertIn("F.cross_entropy", train_source)
        self.assertIn('"t1_yes": T1_THRESHOLD', train_source)
        self.assertIn('"t3_yes": T3_THRESHOLD', train_source)
        self.assertIn('"multiclass_decision": "argmax"', train_source)
        self.assertIn("compute_class_weights", train_source)
        self.assertIn("weight=class_weights[task]", train_source)
        self.assertIn('"ensemble_members": [', train_source)
        self.assertIn("ensemble_probabilities", inference_source)
        self.assertIn("only supports simplified artifact v9", inference_source)

    def test_long_training_uses_differential_learning_rates(self):
        source = builder.TRAIN_CONFIG + builder.TRAIN_CLASSIFIER
        self.assertIn("MLM_EPOCHS = 10", source)
        self.assertIn("MLM_LR = 3e-5", source)
        self.assertIn("RUN_HF_UPLOAD = False", source)
        self.assertIn("MAX_EPOCHS = 30", source)
        self.assertIn("EARLY_STOPPING_PATIENCE = 5", source)
        self.assertIn("LORA_LEARNING_RATE = 5e-5", source)
        self.assertIn('"lr": LORA_LEARNING_RATE', source)
        self.assertIn('"lr": HEAD_LEARNING_RATE', source)
        self.assertIn('"lora_learning_rate": current_lrs[0]', source)
        self.assertIn('"head_learning_rate": current_lrs[1]', source)

    def test_residual_head_matches_inference(self):
        train_source = builder.TRAIN_CLASSIFIER
        inference_source = builder.INFERENCE_MAIN
        expected = [
            "LearnableLayerPool(num_layers=4)",
            "nn.Linear(hidden_size, 512)",
            "ResidualAdapter(512, 256)",
            "ResidualAdapter(512, 128)",
            "nn.Linear(512, num_classes)",
            "nn.init.zeros_(self.up.weight)",
            "output_hidden_states=True",
            "self.layer_pooler(outputs.hidden_states)",
        ]
        for token in expected:
            self.assertIn(token, train_source)
            self.assertIn(token, inference_source)

    def test_last_four_cls_pool_starts_as_equal_average(self):
        components = load_model_components(builder.TRAIN_CLASSIFIER)
        pooler = components["LearnableLayerPool"](num_layers=4)
        hidden_states = [
            torch.randn(2, 3, 8)
            for _ in range(6)
        ]
        actual = pooler(hidden_states)
        expected = torch.stack(
            [layer[:, 0, :] for layer in hidden_states[-4:]],
            dim=0,
        ).mean(dim=0)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(
            torch.allclose(
                torch.softmax(pooler.layer_logits, dim=0),
                torch.full((4,), 0.25),
            )
        )

    def test_shared_and_task_heads_have_expected_shapes(self):
        components = load_model_components(builder.TRAIN_CLASSIFIER)
        adapter = components["ResidualAdapter"](
            feature_size=16,
            bottleneck_size=4,
        )
        adapter_input = torch.randn(3, 16)
        self.assertTrue(torch.equal(adapter(adapter_input), adapter_input))
        self.assertTrue(
            torch.count_nonzero(adapter.up.weight).item() == 0
        )
        self.assertTrue(
            torch.count_nonzero(adapter.up.bias).item() == 0
        )

        shared = components["SharedFeatureMLP"](hidden_size=768)
        heads = {
            task: components["TaskHead"](num_classes)
            for task, num_classes in {
                "t1": 2,
                "t2": 4,
                "t3": 2,
                "t4": 3,
            }.items()
        }
        shared.eval()
        for head in heads.values():
            head.eval()
        features = shared(torch.randn(3, 768))
        self.assertEqual(tuple(features.shape), (3, 512))
        self.assertTrue(torch.isfinite(features).all())
        for task, head in heads.items():
            logits = head(features)
            self.assertEqual(
                tuple(logits.shape),
                (3, head.classifier.out_features),
                task,
            )
            self.assertTrue(torch.isfinite(logits).all(), task)
            self.assertTrue(
                torch.count_nonzero(head.adapter.up.weight).item() == 0,
                task,
            )
            self.assertTrue(
                torch.count_nonzero(head.adapter.up.bias).item() == 0,
                task,
            )
        self.assertTrue(
            torch.count_nonzero(shared.adapter.up.weight).item() == 0
        )
        self.assertTrue(
            torch.count_nonzero(shared.adapter.up.bias).item() == 0
        )

    def test_v9_head_checkpoint_matches_inference(self):
        train = load_model_components(builder.TRAIN_CLASSIFIER)
        inference = load_model_components(builder.INFERENCE_MAIN)

        def build_state(components):
            pooler = components["LearnableLayerPool"](num_layers=4)
            shared = components["SharedFeatureMLP"](hidden_size=768)
            heads = nn.ModuleDict(
                {
                    "t1": components["TaskHead"](2),
                    "t2": components["TaskHead"](4),
                    "t3": components["TaskHead"](2),
                    "t4": components["TaskHead"](3),
                }
            )
            return pooler, shared, heads

        train_modules = build_state(train)
        state = {
            "layer_pooler": train_modules[0].state_dict(),
            "shared_mlp": train_modules[1].state_dict(),
            "heads": train_modules[2].state_dict(),
        }
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)

        inference_modules = build_state(inference)
        inference_modules[0].load_state_dict(restored["layer_pooler"])
        inference_modules[1].load_state_dict(restored["shared_mlp"])
        inference_modules[2].load_state_dict(restored["heads"])

        for train_module, inference_module in zip(
            train_modules,
            inference_modules,
        ):
            train_state = train_module.state_dict()
            inference_state = inference_module.state_dict()
            self.assertEqual(train_state.keys(), inference_state.keys())
            for key in train_state:
                self.assertTrue(
                    torch.equal(train_state[key], inference_state[key]),
                    key,
                )

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
