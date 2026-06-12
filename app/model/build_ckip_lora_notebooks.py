"""Generate the CKIP-BERT LoRA training and inference notebooks.

The notebooks are generated from plain Python strings so their implementation
can be reviewed, compiled, and tested without editing notebook JSON directly.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


TRAIN_IMPORTS_CONFIG = r'''import gc
import json
import math
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, notebook_login
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")


# ==========================================
# 0. Configuration
# ==========================================

MODEL_NAME = "ckiplab/bert-base-chinese"
FOLDS = [1, 2, 3, 4, 5]
FULL_DATA_SEEDS = [42, 123, 2026]
SEED = 42

MAX_LEN = 512
HEAD_RATIO = 0.25
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
MAX_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 3
MIN_SCORE_IMPROVEMENT = 1e-4

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.10
LORA_LR = 1e-4
HEAD_LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0

FOCAL_GAMMA = 1.5
EFFECTIVE_NUMBER_BETA = 0.999
MAX_CLASS_WEIGHT = 6.0
T2_ORDINAL_LOSS_WEIGHT = 0.15
PAIR_LOSS_WEIGHT = 0.05
PAIR_COSINE_MARGIN = 0.50
SYNTHETIC_T4_SAMPLE_WEIGHT = 0.25
SYNTHETIC_ID_MIN = 90000
MISLEADING_MAX_REAL_FPR = 0.005

TASK_WEIGHTS = {
    "t1": 0.20,
    "t2": 0.15,
    "t3": 0.30,
    "t4": 0.35,
}

ID_COLUMN = "id"
TEXT_COLUMN = "data"
TARGET_COLUMNS = [
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]
TASK_COLUMNS = {
    "t1": "promise_status",
    "t2": "verification_timeline",
    "t3": "evidence_status",
    "t4": "evidence_quality",
}
TASK_CLASSES = {
    "t1": ["No", "Yes"],
    "t2": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years"],
    "t3": ["No", "Yes"],
    "t4": ["Clear", "Not Clear", "Misleading"],
}
TASK_LABEL_MAPS = {
    task: {label: index for index, label in enumerate(labels)}
    for task, labels in TASK_CLASSES.items()
}

BASELINE_METRICS = {
    "competition": 0.6011399224663667,
    "promise_status": 0.7543625926016904,
    "verification_timeline": 0.5655582681299120,
    "evidence_status": 0.7249772640915915,
    "evidence_quality": 0.42268709856875536,
    "misleading_false_positive_rate": 11 / 1343,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

PROJECT_ROOT = Path.cwd().resolve()
for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (candidate / "app" / "data" / "clean_data").exists():
        PROJECT_ROOT = candidate
        break

LOCAL_DATA_DIR = PROJECT_ROOT / "app" / "data"
RAW_BASE_URL = "https://raw.githubusercontent.com/Heng1222/VeriPromiseESG_2026_TEAM_9906/feat-model-train/app/data/"

OUTPUT_DIR = Path("mtl_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
OOF_LOGITS_CSV = OUTPUT_DIR / "mtl_oof_logits.csv"
OOF_PROBABILITY_CSV = OUTPUT_DIR / "mtl_oof_probabilities.csv"
OOF_PREDICTION_CSV = OUTPUT_DIR / "mtl_oof_predictions.csv"
SYNTHETIC_HOLDOUT_LOGITS_CSV = OUTPUT_DIR / "mtl_synthetic_holdout_logits.csv"
CALIBRATION_JSON = OUTPUT_DIR / "mtl_calibration.json"
THRESHOLD_JSON = OUTPUT_DIR / "mtl_thresholds.json"
INFERENCE_CONFIG_JSON = OUTPUT_DIR / "mtl_inference_config.json"

RUN_HF_UPLOAD = False
HF_MTL_REPO_ID = "maxbeettww/VeriPromise_ESG_2026_9906"
HF_PRIVATE_REPO = False
HF_COMMIT_MESSAGE = "Upload CKIP-BERT LoRA MTL artifact v3"

print(f"Device: {DEVICE}")
print(f"Project root: {PROJECT_ROOT}")
'''


TRAIN_DATA = r'''# ==========================================
# 1. Data loading, normalization, and synthetic pairing
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_value(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.upper() == "N/A":
        return None
    if value == "longer_than_5_years":
        return "more_than_5_years"
    return value


def read_csv_local_or_remote(relative_path):
    local_path = LOCAL_DATA_DIR / relative_path
    if local_path.exists():
        return pd.read_csv(local_path)
    return pd.read_csv(f"{RAW_BASE_URL}{relative_path.as_posix()}")


def read_fold_csv(fold, split):
    if fold not in FOLDS:
        raise ValueError(f"Unknown fold: {fold}")
    if split not in {"train", "val"}:
        raise ValueError(f"Unknown split: {split}")
    return read_csv_local_or_remote(
        Path("clean_data") / f"{split}_fold_{fold}.csv"
    )


def normalize_training_frame(df):
    result = df.copy()
    for column in TARGET_COLUMNS:
        result[column] = result[column].apply(normalize_value)
    return result


def load_real_data():
    """Load the five prepared train/validation folds.

    Validation folds are disjoint and together contain all 2,000 real rows.
    The same synthetic rows are injected into every train fold, so they are
    de-duplicated by id before synthetic source-group assignment.
    """
    fold_frames = {}
    real_parts = []
    synthetic_parts = []
    for fold in FOLDS:
        train_df = normalize_training_frame(read_fold_csv(fold, "train"))
        val_df = normalize_training_frame(read_fold_csv(fold, "val"))
        validate_training_frame(train_df, f"train_fold_{fold}")
        validate_training_frame(val_df, f"val_fold_{fold}")

        numeric_train_ids = pd.to_numeric(
            train_df[ID_COLUMN],
            errors="coerce",
        )
        train_real_df = train_df[numeric_train_ids < SYNTHETIC_ID_MIN].copy()
        train_synthetic_df = train_df[
            numeric_train_ids >= SYNTHETIC_ID_MIN
        ].copy()
        if set(train_real_df[ID_COLUMN]) & set(val_df[ID_COLUMN]):
            raise ValueError(f"Fold {fold} has train/validation id leakage.")

        fold_frames[fold] = {
            "train": train_df,
            "train_real": train_real_df,
            "val": val_df,
        }
        real_parts.append(val_df)
        synthetic_parts.append(train_synthetic_df)

    real_df = pd.concat(real_parts, ignore_index=True)
    validate_training_frame(real_df, "combined_real_data")
    numeric_real_ids = pd.to_numeric(real_df[ID_COLUMN], errors="coerce")
    if numeric_real_ids.isna().any() or (
        numeric_real_ids >= SYNTHETIC_ID_MIN
    ).any():
        raise ValueError("Validation folds must contain only real numeric ids.")

    synthetic_all = pd.concat(synthetic_parts, ignore_index=True)
    synthetic_df = synthetic_all.drop_duplicates(
        subset=[ID_COLUMN],
        keep="first",
    ).reset_index(drop=True)
    validate_training_frame(synthetic_df, "combined_synthetic_data")
    expected_synthetic_ids = set(synthetic_df[ID_COLUMN])
    for fold in FOLDS:
        fold_synthetic_ids = set(
            fold_frames[fold]["train"][ID_COLUMN]
        ) - set(fold_frames[fold]["train_real"][ID_COLUMN])
        if fold_synthetic_ids != expected_synthetic_ids:
            raise ValueError(
                f"Fold {fold} synthetic ids differ from the other folds."
            )

    print(
        f"Loaded five folds: real={len(real_df)}, "
        f"synthetic={len(synthetic_df)}"
    )
    return real_df, synthetic_df, fold_frames


def validate_training_frame(df, name):
    required = [ID_COLUMN, TEXT_COLUMN] + TARGET_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError(f"{name} contains duplicated ids.")


def load_fold_ids(fold):
    fold_df = read_fold_csv(fold, "val")
    return set(fold_df[ID_COLUMN].tolist())


def source_group_key(df):
    return (
        df["pdf_url"].fillna("").astype(str).str.strip()
        + "||"
        + df["page_number"].fillna("").astype(str).str.strip()
    )


def assign_synthetic_folds(synthetic_df, n_splits=5, seed=SEED):
    groups = sorted(source_group_key(synthetic_df).unique().tolist())
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    assignment = {group: (index % n_splits) + 1 for index, group in enumerate(groups)}
    result = synthetic_df.copy()
    result["synthetic_fold"] = source_group_key(result).map(assignment).astype(int)
    return result


def attach_source_pairs(synthetic_df, real_df):
    source_lookup = (
        real_df.loc[
            real_df["evidence_quality"].eq("Clear"),
            ["pdf_url", "promise_string", TEXT_COLUMN],
        ]
        .dropna(subset=["pdf_url", "promise_string", TEXT_COLUMN])
        .drop_duplicates(["pdf_url", "promise_string"])
        .rename(columns={TEXT_COLUMN: "pair_source_text"})
    )
    paired = synthetic_df.merge(
        source_lookup,
        how="left",
        on=["pdf_url", "promise_string"],
        validate="many_to_one",
    )
    paired["pair_valid"] = paired["pair_source_text"].notna()
    print(
        f"Paired synthetic rows: {int(paired['pair_valid'].sum())} / {len(paired)}"
    )
    return paired


def build_fold_frames(fold_frames, synthetic_df, fold):
    val_df = fold_frames[fold]["val"].copy()
    train_real_df = fold_frames[fold]["train_real"].copy()
    synthetic_train = synthetic_df[synthetic_df["synthetic_fold"] != fold].copy()
    synthetic_holdout = synthetic_df[synthetic_df["synthetic_fold"] == fold].copy()
    train_df = pd.concat([train_real_df, synthetic_train], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=SEED + fold).reset_index(drop=True)
    validate_training_frame(train_df, f"train_fold_{fold}")
    validate_training_frame(val_df, f"val_fold_{fold}")
    return train_df, val_df, synthetic_holdout


def tokenize_head_tail(text, tokenizer, max_len=MAX_LEN, head_ratio=HEAD_RATIO):
    body_ids = tokenizer.encode(
        f"文本：{str(text)}",
        add_special_tokens=False,
        verbose=False,
    )
    max_body_len = max_len - 2
    if len(body_ids) > max_body_len:
        head_len = int(max_body_len * head_ratio)
        tail_len = max_body_len - head_len
        body_ids = body_ids[:head_len] + body_ids[-tail_len:]
    input_ids = [tokenizer.cls_token_id] + body_ids + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    pad_len = max_len - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len
    return input_ids, attention_mask


class ESGMTLDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=MAX_LEN, is_test=False):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        input_ids, attention_mask = tokenize_head_tail(
            row[TEXT_COLUMN], self.tokenizer, self.max_len
        )
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "row_index": torch.tensor(index, dtype=torch.long),
        }
        if self.is_test:
            return item

        numeric_id = pd.to_numeric(row[ID_COLUMN], errors="coerce")
        is_synthetic = bool(pd.notna(numeric_id) and numeric_id >= SYNTHETIC_ID_MIN)
        for task, column in TASK_COLUMNS.items():
            value = normalize_value(row.get(column))
            item[f"{task}_label"] = torch.tensor(
                TASK_LABEL_MAPS[task].get(value, -1),
                dtype=torch.long,
            )
        item["is_synthetic"] = torch.tensor(is_synthetic, dtype=torch.bool)

        pair_valid = bool(row.get("pair_valid", False)) and is_synthetic
        pair_text = row.get("pair_source_text") if pair_valid else row[TEXT_COLUMN]
        pair_ids, pair_mask = tokenize_head_tail(
            pair_text, self.tokenizer, self.max_len
        )
        item["pair_valid"] = torch.tensor(pair_valid, dtype=torch.bool)
        item["pair_input_ids"] = torch.tensor(pair_ids, dtype=torch.long)
        item["pair_attention_mask"] = torch.tensor(pair_mask, dtype=torch.long)
        return item
'''


TRAIN_MODEL = r'''# ==========================================
# 2. LoRA backbone, pooling, MLP heads, and losses
# ==========================================

def make_lora_config():
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        bias="none",
        use_rslora=True,
    )


class ESGLoraMTLModel(nn.Module):
    def __init__(self, model_name, adapter_dir=None, is_trainable=True):
        super().__init__()
        base_model = AutoModel.from_pretrained(model_name)
        if adapter_dir is None:
            self.backbone = get_peft_model(base_model, make_lora_config())
        else:
            self.backbone = PeftModel.from_pretrained(
                base_model,
                adapter_dir,
                is_trainable=is_trainable,
            )
        hidden_size = base_model.config.hidden_size
        pooled_size = hidden_size * 3
        self.shared_mlp = nn.Sequential(
            nn.Linear(pooled_size, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.multi_sample_dropouts = nn.ModuleList(
            [nn.Dropout(probability) for probability in [0.10, 0.20, 0.30, 0.40, 0.50]]
        )
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(256, 128),
                    nn.LayerNorm(128),
                    nn.GELU(),
                    nn.Dropout(0.20),
                    nn.Linear(128, len(labels)),
                )
                for task, labels in TASK_CLASSES.items()
            }
        )

    def encode(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        token_mask = attention_mask.unsqueeze(-1).bool()
        cls_pool = hidden[:, 0, :]
        mean_pool = (hidden * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        max_pool = hidden.masked_fill(~token_mask, torch.finfo(hidden.dtype).min).max(dim=1).values
        pooled = torch.cat([cls_pool, mean_pool, max_pool], dim=-1)
        return self.shared_mlp(pooled)

    def forward(self, input_ids, attention_mask, return_features=False):
        features = self.encode(input_ids, attention_mask)
        logits = {
            task: torch.stack(
                [head(dropout(features)) for dropout in self.multi_sample_dropouts],
                dim=0,
            ).mean(dim=0)
            for task, head in self.heads.items()
        }
        if return_features:
            return logits, features
        return logits

    def head_state_dict(self):
        return {
            "shared_mlp": self.shared_mlp.state_dict(),
            "heads": self.heads.state_dict(),
        }

    def load_head_state_dict(self, state):
        self.shared_mlp.load_state_dict(state["shared_mlp"])
        self.heads.load_state_dict(state["heads"])


def effective_number_weights(counts):
    counts = np.asarray(counts, dtype=np.float64)
    effective = (1.0 - np.power(EFFECTIVE_NUMBER_BETA, counts)) / (
        1.0 - EFFECTIVE_NUMBER_BETA
    )
    weights = 1.0 / np.maximum(effective, 1e-8)
    weights = weights / weights.mean()
    weights = np.minimum(weights, MAX_CLASS_WEIGHT)
    return torch.tensor(weights, dtype=torch.float, device=DEVICE)


def compute_fold_class_weights(train_df):
    work = train_df.copy()
    synthetic = pd.to_numeric(work[ID_COLUMN], errors="coerce") >= SYNTHETIC_ID_MIN
    output = {}
    for task, column in TASK_COLUMNS.items():
        counts = []
        for label in TASK_CLASSES[task]:
            real_count = int((work[column].eq(label) & ~synthetic).sum())
            synthetic_count = int((work[column].eq(label) & synthetic).sum())
            if task == "t4":
                counts.append(
                    real_count + SYNTHETIC_T4_SAMPLE_WEIGHT * synthetic_count
                )
            else:
                counts.append(real_count)
        output[task] = effective_number_weights(counts)
    return output


def focal_cross_entropy(logits, labels, class_weights):
    ce = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        reduction="none",
    )
    probabilities = torch.softmax(logits, dim=-1)
    true_probability = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    return ((1.0 - true_probability).pow(FOCAL_GAMMA) * ce)


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def calculate_mtl_loss(model, logits, features, batch, class_weights):
    labels = {
        task: batch[f"{task}_label"].to(DEVICE)
        for task in TASK_CLASSES
    }
    is_synthetic = batch["is_synthetic"].to(DEVICE)
    real_mask = ~is_synthetic
    zero = features.sum() * 0.0
    losses = {task: zero for task in TASK_CLASSES}

    masks = {
        "t1": real_mask & (labels["t1"] >= 0),
        "t2": real_mask & (labels["t1"] == 1) & (labels["t2"] >= 0),
        "t3": real_mask & (labels["t1"] == 1) & (labels["t3"] >= 0),
        "t4": (labels["t1"] == 1) & (labels["t3"] == 1) & (labels["t4"] >= 0),
    }
    for task, valid in masks.items():
        if not valid.any():
            continue
        element_loss = focal_cross_entropy(
            logits[task][valid],
            labels[task][valid],
            class_weights[task],
        )
        sample_weights = torch.ones_like(element_loss)
        if task == "t4":
            sample_weights = torch.where(
                is_synthetic[valid],
                torch.full_like(sample_weights, SYNTHETIC_T4_SAMPLE_WEIGHT),
                sample_weights,
            )
        losses[task] = weighted_mean(element_loss, sample_weights)

    t2_valid = masks["t2"]
    if t2_valid.any():
        positions = torch.arange(
            len(TASK_CLASSES["t2"]),
            dtype=logits["t2"].dtype,
            device=DEVICE,
        )
        expected_position = (
            torch.softmax(logits["t2"][t2_valid], dim=-1) * positions
        ).sum(dim=-1)
        target_position = labels["t2"][t2_valid].to(expected_position.dtype)
        ordinal_distance = (expected_position - target_position).abs() / (
            len(TASK_CLASSES["t2"]) - 1
        )
        ordinal_weights = class_weights["t2"][labels["t2"][t2_valid]]
        losses["t2"] = losses["t2"] + T2_ORDINAL_LOSS_WEIGHT * weighted_mean(
            ordinal_distance,
            ordinal_weights,
        )

    pair_loss = zero
    pair_valid_cpu = batch["pair_valid"].bool()
    if pair_valid_cpu.any():
        pair_valid = pair_valid_cpu.to(DEVICE)
        pair_features = model.encode(
            batch["pair_input_ids"][pair_valid_cpu].to(DEVICE),
            batch["pair_attention_mask"][pair_valid_cpu].to(DEVICE),
        )
        similarity = F.cosine_similarity(features[pair_valid], pair_features)
        pair_loss = F.relu(similarity - PAIR_COSINE_MARGIN).mean()

    total = sum(TASK_WEIGHTS[task] * losses[task] for task in TASK_CLASSES)
    total = total + PAIR_LOSS_WEIGHT * pair_loss
    return total, losses, pair_loss
'''


TRAIN_METRICS = r'''# ==========================================
# 3. Calibration, routing, metrics, and diagnostics
# ==========================================

def softmax_numpy(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def predict_logits(model, data_loader):
    model.eval()
    output = {task: [] for task in TASK_CLASSES}
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                logits = model(input_ids, attention_mask)
            for task in TASK_CLASSES:
                output[task].append(logits[task].float().cpu().numpy())
    return {
        task: np.concatenate(parts, axis=0)
        for task, parts in output.items()
    }


def logits_to_frame(df, logits, prefix=""):
    output = pd.DataFrame({ID_COLUMN: df[ID_COLUMN].values})
    for task, labels in TASK_CLASSES.items():
        for index, label in enumerate(labels):
            output[f"{prefix}{task}__{label}"] = logits[task][:, index]
    return output


def frame_to_logits(frame, prefix=""):
    return {
        task: frame[
            [f"{prefix}{task}__{label}" for label in labels]
        ].to_numpy()
        for task, labels in TASK_CLASSES.items()
    }


def fit_scalar_temperatures(true_df, logits_df):
    temperatures = {}
    merged = true_df[[ID_COLUMN] + TARGET_COLUMNS].merge(
        logits_df,
        on=ID_COLUMN,
        validate="one_to_one",
    )
    grid = np.linspace(0.50, 3.00, 251)
    for task, column in TASK_COLUMNS.items():
        valid = merged[column].notna()
        labels = merged.loc[valid, column].map(TASK_LABEL_MAPS[task]).astype(int).to_numpy()
        logits = merged.loc[
            valid,
            [f"{task}__{label}" for label in TASK_CLASSES[task]],
        ].to_numpy()
        best_temperature = 1.0
        best_nll = float("inf")
        for temperature in grid:
            probabilities = softmax_numpy(logits / temperature)
            nll = -np.log(
                probabilities[np.arange(len(labels)), labels].clip(1e-12, 1.0)
            ).mean()
            if nll < best_nll:
                best_nll = float(nll)
                best_temperature = float(temperature)
        temperatures[task] = best_temperature
    return temperatures


def calibrated_probabilities(logits, temperatures):
    return {
        task: softmax_numpy(logits[task] / float(temperatures.get(task, 1.0)))
        for task in TASK_CLASSES
    }


def probabilities_to_frame(df, probabilities):
    output = pd.DataFrame({ID_COLUMN: df[ID_COLUMN].values})
    for task, labels in TASK_CLASSES.items():
        for index, label in enumerate(labels):
            output[f"{task}__{label}"] = probabilities[task][:, index]
    return output


def argmax_label(row, task, labels=None):
    labels = labels or TASK_CLASSES[task]
    values = [row[f"{task}__{label}"] for label in labels]
    return labels[int(np.argmax(values))]


def choose_t4_label(row, misleading_thresholds):
    standard_labels = ["Clear", "Not Clear"]
    standard_label = argmax_label(row, "t4", standard_labels)
    standard_probability = row[f"t4__{standard_label}"]
    misleading_probability = row["t4__Misleading"]
    probability_threshold = misleading_thresholds.get(
        "probability_threshold", 1.0
    )
    margin_threshold = misleading_thresholds.get("margin_threshold", 1.0)
    if (
        misleading_probability >= probability_threshold
        and misleading_probability - standard_probability >= margin_threshold
    ):
        return "Misleading"
    return standard_label


def route_predictions(
    probability_df,
    t1_threshold=0.5,
    t3_threshold=0.5,
    misleading_thresholds=None,
):
    misleading_thresholds = misleading_thresholds or {}
    results = []
    for row_index, row in probability_df.iterrows():
        source_id = probability_df.at[row_index, ID_COLUMN]
        t1_prediction = (
            "Yes" if row["t1__Yes"] >= t1_threshold else "No"
        )
        if t1_prediction == "No":
            results.append(
                {
                    ID_COLUMN: source_id,
                    "promise_status": "No",
                    "verification_timeline": "N/A",
                    "evidence_status": "N/A",
                    "evidence_quality": "N/A",
                }
            )
            continue
        t2_prediction = argmax_label(row, "t2")
        t3_prediction = (
            "Yes" if row["t3__Yes"] >= t3_threshold else "No"
        )
        if t3_prediction == "No":
            results.append(
                {
                    ID_COLUMN: source_id,
                    "promise_status": "Yes",
                    "verification_timeline": t2_prediction,
                    "evidence_status": "No",
                    "evidence_quality": "N/A",
                }
            )
            continue
        results.append(
            {
                ID_COLUMN: source_id,
                "promise_status": "Yes",
                "verification_timeline": t2_prediction,
                "evidence_status": "Yes",
                "evidence_quality": choose_t4_label(
                    row,
                    misleading_thresholds,
                ),
            }
        )
    return pd.DataFrame(results)[[ID_COLUMN] + TARGET_COLUMNS]


def strict_task_f1(true_df, prediction_df, column):
    merged = true_df[[ID_COLUMN, column]].merge(
        prediction_df[[ID_COLUMN, column]],
        on=ID_COLUMN,
        suffixes=("_true", "_pred"),
        validate="one_to_one",
    )
    y_true = merged[f"{column}_true"].apply(normalize_value)
    y_pred = merged[f"{column}_pred"].apply(normalize_value)
    valid = y_true.notna()
    task = {column: task for task, column in TASK_COLUMNS.items()}[column]
    return f1_score(
        y_true[valid],
        y_pred[valid].fillna("N/A"),
        labels=TASK_CLASSES[task],
        average="macro",
        zero_division=0,
    )


def evaluate_submission(true_df, prediction_df):
    scores = {
        column: strict_task_f1(true_df, prediction_df, column)
        for column in TARGET_COLUMNS
    }
    competition_score = sum(
        TASK_WEIGHTS[task] * scores[column]
        for task, column in TASK_COLUMNS.items()
    )
    rows = [
        {"task": column, "macro_f1": score}
        for column, score in scores.items()
    ]
    rows.append({"task": "competition", "macro_f1": competition_score})
    return pd.DataFrame(rows)


def competition_score(true_df, prediction_df):
    metrics = evaluate_submission(true_df, prediction_df)
    return float(
        metrics.loc[metrics["task"] == "competition", "macro_f1"].iloc[0]
    )


def tune_routing_thresholds(true_df, probability_df):
    fold_ids = {
        fold: load_fold_ids(fold)
        for fold in FOLDS
    }
    best = {
        "objective": -1.0,
        "t1_threshold": 0.5,
        "t3_threshold": 0.5,
    }
    grid = np.round(np.arange(0.25, 0.701, 0.02), 2)
    for t1_threshold in grid:
        for t3_threshold in grid:
            prediction_df = route_predictions(
                probability_df,
                t1_threshold=t1_threshold,
                t3_threshold=t3_threshold,
            )
            fold_scores = []
            for fold in FOLDS:
                ids = fold_ids[fold]
                fold_true = true_df[true_df[ID_COLUMN].isin(ids)]
                fold_prediction = prediction_df[
                    prediction_df[ID_COLUMN].isin(ids)
                ]
                fold_scores.append(
                    competition_score(fold_true, fold_prediction)
                )
            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))
            objective = mean_score - 0.25 * std_score
            if objective > best["objective"]:
                best = {
                    "objective": objective,
                    "mean_fold_score": mean_score,
                    "fold_score_std": std_score,
                    "fold_scores": fold_scores,
                    "t1_threshold": float(t1_threshold),
                    "t3_threshold": float(t3_threshold),
                }
    return best


def misleading_mask(probability_df, probability_threshold, margin_threshold):
    standard_max = probability_df[
        ["t4__Clear", "t4__Not Clear"]
    ].max(axis=1)
    misleading_probability = probability_df["t4__Misleading"]
    return (
        (misleading_probability >= probability_threshold)
        & (
            misleading_probability - standard_max
            >= margin_threshold
        )
    )


def tune_misleading_thresholds(
    real_true_df,
    real_probability_df,
    synthetic_probability_df,
):
    merged = real_true_df[
        [ID_COLUMN, "evidence_quality"]
    ].merge(real_probability_df, on=ID_COLUMN, validate="one_to_one")
    valid_real = merged["evidence_quality"].notna()
    non_misleading = valid_real & ~merged["evidence_quality"].eq("Misleading")
    true_misleading = valid_real & merged["evidence_quality"].eq("Misleading")

    best = None
    probability_grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    margin_grid = np.round(np.arange(-0.20, 0.301, 0.01), 2)
    for probability_threshold in probability_grid:
        for margin_threshold in margin_grid:
            real_predictions = misleading_mask(
                merged,
                probability_threshold,
                margin_threshold,
            )
            false_positive_rate = float(
                real_predictions[non_misleading].mean()
            )
            if false_positive_rate > MISLEADING_MAX_REAL_FPR:
                continue
            synthetic_predictions = misleading_mask(
                synthetic_probability_df,
                probability_threshold,
                margin_threshold,
            )
            synthetic_recall = float(synthetic_predictions.mean())
            real_recall = (
                float(real_predictions[true_misleading].mean())
                if true_misleading.any()
                else 0.0
            )
            candidate = {
                "probability_threshold": float(probability_threshold),
                "margin_threshold": float(margin_threshold),
                "real_false_positive_rate": false_positive_rate,
                "synthetic_holdout_recall": synthetic_recall,
                "real_misleading_recall": real_recall,
            }
            rank = (
                synthetic_recall,
                real_recall,
                -false_positive_rate,
                probability_threshold,
                margin_threshold,
            )
            if best is None or rank > best[0]:
                best = (rank, candidate)
    if best is None:
        return {
            "probability_threshold": 1.0,
            "margin_threshold": 1.0,
            "real_false_positive_rate": 0.0,
            "synthetic_holdout_recall": 0.0,
            "real_misleading_recall": 0.0,
        }
    return best[1]


def print_reports(true_df, prediction_df, title):
    print(f"\n{'=' * 24} {title} {'=' * 24}")
    display(evaluate_submission(true_df, prediction_df))
    print("Prediction distributions:")
    for column in TARGET_COLUMNS:
        print(column, prediction_df[column].value_counts(dropna=False).to_dict())
    for task, column in TASK_COLUMNS.items():
        merged = true_df[[ID_COLUMN, column]].merge(
            prediction_df[[ID_COLUMN, column]],
            on=ID_COLUMN,
            suffixes=("_true", "_pred"),
        )
        y_true = merged[f"{column}_true"].apply(normalize_value)
        y_pred = merged[f"{column}_pred"].apply(normalize_value)
        valid = y_true.notna()
        print(f"\n=== {column} ===")
        print(
            classification_report(
                y_true[valid],
                y_pred[valid].fillna("N/A"),
                labels=TASK_CLASSES[task],
                zero_division=0,
            )
        )
        print(
            pd.DataFrame(
                confusion_matrix(
                    y_true[valid],
                    y_pred[valid].fillna("N/A"),
                    labels=TASK_CLASSES[task],
                ),
                index=TASK_CLASSES[task],
                columns=TASK_CLASSES[task],
            )
        )
'''


TRAIN_ARTIFACTS = r'''# ==========================================
# 4. Optimizer and artifact helpers
# ==========================================

def create_optimizer(model):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    groups = []
    modules = [
        (model.backbone, LORA_LR),
        (nn.ModuleList([model.shared_mlp, model.heads]), HEAD_LR),
    ]
    for module, learning_rate in modules:
        named_parameters = [
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ]
        for apply_decay in [True, False]:
            parameters = [
                parameter
                for name, parameter in named_parameters
                if (not any(token in name for token in no_decay)) == apply_decay
            ]
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "lr": learning_rate,
                        "weight_decay": WEIGHT_DECAY if apply_decay else 0.0,
                    }
                )
    return torch.optim.AdamW(groups)


def save_v3_model(model, member_dir, metadata):
    member_dir = Path(member_dir)
    adapter_dir = member_dir / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(adapter_dir, safe_serialization=True)
    torch.save(model.head_state_dict(), member_dir / "heads.pt")
    with open(member_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def load_v3_model(member_dir, is_trainable=False):
    member_dir = Path(member_dir)
    model = ESGLoraMTLModel(
        MODEL_NAME,
        adapter_dir=member_dir / "adapter",
        is_trainable=is_trainable,
    ).to(DEVICE)
    head_state = torch.load(
        member_dir / "heads.pt",
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_head_state_dict(head_state)
    return model


def assert_save_load_parity(model, member_dir, data_loader, atol=1e-5):
    batch = next(iter(data_loader))
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    model.eval()
    with torch.no_grad():
        expected = model(input_ids, attention_mask)
    reloaded = load_v3_model(member_dir)
    reloaded.eval()
    with torch.no_grad():
        actual = reloaded(input_ids, attention_mask)
    for task in TASK_CLASSES:
        torch.testing.assert_close(
            actual[task],
            expected[task],
            rtol=0.0,
            atol=atol,
        )
    del reloaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Save/load parity passed for {member_dir}.")


def make_loader(
    dataframe,
    tokenizer,
    shuffle=False,
    seed=SEED,
    is_test=False,
):
    dataset = ESGMTLDataset(dataframe, tokenizer, is_test=is_test)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=USE_AMP,
    )


def train_epochs(
    model,
    train_loader,
    class_weights,
    epochs,
    validation_callback=None,
    checkpoint_dir=None,
    patience=None,
):
    optimizer = create_optimizer(model)
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / GRAD_ACCUM_STEPS
    )
    total_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    best_score = -1.0
    best_epoch = 0
    without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        pair_loss_sum = 0.0
        task_loss_sums = {task: 0.0 for task in TASK_CLASSES}
        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"),
            start=1,
        ):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                logits, features = model(
                    input_ids,
                    attention_mask,
                    return_features=True,
                )
                loss, task_losses, pair_loss = calculate_mtl_loss(
                    model,
                    logits,
                    features,
                    batch,
                    class_weights,
                )
                scaled_loss = loss / GRAD_ACCUM_STEPS
            scaler.scale(scaled_loss).backward()
            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach().cpu())
            pair_loss_sum += float(pair_loss.detach().cpu())
            for task in TASK_CLASSES:
                task_loss_sums[task] += float(
                    task_losses[task].detach().cpu()
                )

        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train_loader),
            "pair_loss": pair_loss_sum / len(train_loader),
        }
        for task in TASK_CLASSES:
            row[f"{task}_loss"] = (
                task_loss_sums[task] / len(train_loader)
            )

        score = None
        metrics = None
        if validation_callback is not None:
            score, metrics = validation_callback(model)
            row.update(metrics)
            row["competition_score"] = score
        history.append(row)
        print(pd.DataFrame([row]).to_string(index=False))

        if validation_callback is None:
            continue
        if score > best_score + MIN_SCORE_IMPROVEMENT:
            best_score = score
            best_epoch = epoch
            without_improvement = 0
            save_v3_model(
                model,
                checkpoint_dir,
                {
                    "artifact_version": 3,
                    "best_epoch": epoch,
                    "metrics": metrics,
                },
            )
        else:
            without_improvement += 1
            if patience is not None and without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    return pd.DataFrame(history), best_epoch, best_score
'''


TRAIN_LOOPS = r'''# ==========================================
# 5. Five-fold OOF training
# ==========================================

def train_one_fold(
    fold,
    tokenizer,
    synthetic_df,
    fold_frames,
):
    set_seed(SEED + fold)
    train_df, val_df, synthetic_holdout = build_fold_frames(
        fold_frames,
        synthetic_df,
        fold,
    )
    train_loader = make_loader(
        train_df,
        tokenizer,
        shuffle=True,
        seed=SEED + fold,
    )
    val_loader = make_loader(val_df, tokenizer)
    synthetic_loader = make_loader(
        synthetic_holdout,
        tokenizer,
        is_test=True,
    )
    model = ESGLoraMTLModel(MODEL_NAME).to(DEVICE)
    model.backbone.print_trainable_parameters()
    class_weights = compute_fold_class_weights(train_df)
    fold_dir = OUTPUT_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    def validate(model_to_evaluate):
        logits = predict_logits(model_to_evaluate, val_loader)
        probabilities = {
            task: softmax_numpy(values)
            for task, values in logits.items()
        }
        probability_df = probabilities_to_frame(val_df, probabilities)
        prediction_df = route_predictions(probability_df)
        metrics_df = evaluate_submission(val_df, prediction_df)
        metrics = {
            f"{row.task}_macro_f1": float(row.macro_f1)
            for row in metrics_df.itertuples(index=False)
        }
        return metrics["competition_macro_f1"], metrics

    history, best_epoch, best_score = train_epochs(
        model,
        train_loader,
        class_weights,
        MAX_EPOCHS,
        validation_callback=validate,
        checkpoint_dir=fold_dir,
        patience=EARLY_STOPPING_PATIENCE,
    )
    history.insert(0, "fold", fold)
    history.to_csv(fold_dir / "training_history.csv", index=False)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_model = load_v3_model(fold_dir)
    val_logits = predict_logits(best_model, val_loader)
    synthetic_logits = predict_logits(best_model, synthetic_loader)
    val_logits_df = logits_to_frame(val_df, val_logits)
    val_logits_df["fold"] = fold
    synthetic_logits_df = logits_to_frame(
        synthetic_holdout,
        synthetic_logits,
    )
    synthetic_logits_df["fold"] = fold
    synthetic_logits_df["source_group"] = source_group_key(
        synthetic_holdout
    ).values
    del best_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        val_df[[ID_COLUMN] + TARGET_COLUMNS].copy(),
        val_logits_df,
        synthetic_logits_df,
        history,
        best_epoch,
        best_score,
    )


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.save_pretrained(TOKENIZER_DIR)
real_df, raw_synthetic_df, fold_frames = load_real_data()
synthetic_df = assign_synthetic_folds(
    attach_source_pairs(raw_synthetic_df, real_df)
)
validate_training_frame(real_df, "real_data")
validate_training_frame(synthetic_df, "synthetic_data")

oof_truth_parts = []
oof_logits_parts = []
synthetic_logits_parts = []
history_parts = []
best_epochs = []

for fold in FOLDS:
    print(f"\n{'=' * 64}\nTraining fold {fold}\n{'=' * 64}")
    (
        fold_truth,
        fold_logits,
        fold_synthetic_logits,
        fold_history,
        best_epoch,
        best_score,
    ) = train_one_fold(fold, tokenizer, synthetic_df, fold_frames)
    oof_truth_parts.append(fold_truth)
    oof_logits_parts.append(fold_logits)
    synthetic_logits_parts.append(fold_synthetic_logits)
    history_parts.append(fold_history)
    best_epochs.append(best_epoch)

oof_true_df = pd.concat(oof_truth_parts, ignore_index=True)
oof_logits_df = pd.concat(oof_logits_parts, ignore_index=True)
synthetic_holdout_logits_df = pd.concat(
    synthetic_logits_parts,
    ignore_index=True,
)
training_history_df = pd.concat(history_parts, ignore_index=True)

oof_logits_df.to_csv(OOF_LOGITS_CSV, index=False)
synthetic_holdout_logits_df.to_csv(
    SYNTHETIC_HOLDOUT_LOGITS_CSV,
    index=False,
)
training_history_df.to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False,
)
print(f"Best epochs: {best_epochs}")
'''


TRAIN_CALIBRATE = r'''# ==========================================
# 6. OOF calibration, stable routing, diagnostics, and quality gate
# ==========================================

temperatures = fit_scalar_temperatures(oof_true_df, oof_logits_df)
with open(CALIBRATION_JSON, "w", encoding="utf-8") as file:
    json.dump({"temperatures": temperatures}, file, indent=2)

oof_probabilities = calibrated_probabilities(
    frame_to_logits(oof_logits_df),
    temperatures,
)
oof_probability_df = probabilities_to_frame(
    oof_logits_df,
    oof_probabilities,
)
oof_probability_df["fold"] = oof_logits_df["fold"].values
oof_probability_df.to_csv(OOF_PROBABILITY_CSV, index=False)

synthetic_probabilities = calibrated_probabilities(
    frame_to_logits(synthetic_holdout_logits_df),
    temperatures,
)
synthetic_probability_df = probabilities_to_frame(
    synthetic_holdout_logits_df,
    synthetic_probabilities,
)

routing_thresholds = tune_routing_thresholds(
    oof_true_df,
    oof_probability_df,
)
misleading_thresholds = tune_misleading_thresholds(
    oof_true_df,
    oof_probability_df,
    synthetic_probability_df,
)
thresholds = {
    **routing_thresholds,
    "misleading": misleading_thresholds,
}
with open(THRESHOLD_JSON, "w", encoding="utf-8") as file:
    json.dump(thresholds, file, ensure_ascii=False, indent=2)
print(json.dumps(thresholds, ensure_ascii=False, indent=2))

oof_prediction_df = route_predictions(
    oof_probability_df,
    t1_threshold=thresholds["t1_threshold"],
    t3_threshold=thresholds["t3_threshold"],
    misleading_thresholds=thresholds["misleading"],
)
oof_prediction_df.to_csv(OOF_PREDICTION_CSV, index=False)

print_reports(oof_true_df, oof_prediction_df, "Overall OOF")
print_reports(
    oof_true_df[oof_true_df[ID_COLUMN].between(10001, 11000)],
    oof_prediction_df[oof_prediction_df[ID_COLUMN].between(10001, 11000)],
    "Official train 1000",
)
print_reports(
    oof_true_df[oof_true_df[ID_COLUMN].between(11001, 12000)],
    oof_prediction_df[oof_prediction_df[ID_COLUMN].between(11001, 12000)],
    "Official val 1000",
)
for fold in FOLDS:
    ids = load_fold_ids(fold)
    print_reports(
        oof_true_df[oof_true_df[ID_COLUMN].isin(ids)],
        oof_prediction_df[oof_prediction_df[ID_COLUMN].isin(ids)],
        f"Fold {fold}",
    )

direct_t4_df = pd.DataFrame(
    {
        ID_COLUMN: oof_probability_df[ID_COLUMN],
        "evidence_quality": [
            choose_t4_label(row, thresholds["misleading"])
            for _, row in oof_probability_df.iterrows()
        ],
    }
)
direct_t4_score = strict_task_f1(
    oof_true_df,
    direct_t4_df,
    "evidence_quality",
)
print(f"Direct T4 macro-F1: {direct_t4_score:.6f}")
print("Real Misleading rows:")
display(
    oof_true_df[oof_true_df["evidence_quality"].eq("Misleading")]
    .merge(oof_probability_df, on=ID_COLUMN)
    .merge(
        oof_prediction_df[[ID_COLUMN, "evidence_quality"]],
        on=ID_COLUMN,
        suffixes=("_true", "_pred"),
    )
)
print(
    "Synthetic holdout recall:",
    thresholds["misleading"]["synthetic_holdout_recall"],
)
print(
    "Real non-Misleading false-positive rate:",
    thresholds["misleading"]["real_false_positive_rate"],
)


def run_quality_gate(true_df, prediction_df, misleading_config):
    metrics_df = evaluate_submission(true_df, prediction_df)
    metrics = {
        str(row.task): float(row.macro_f1)
        for row in metrics_df.itertuples(index=False)
    }
    failures = []
    if metrics["competition"] < BASELINE_METRICS["competition"]:
        failures.append(
            f"competition {metrics['competition']:.6f} < "
            f"{BASELINE_METRICS['competition']:.6f}"
        )
    for column in [
        "promise_status",
        "verification_timeline",
        "evidence_status",
    ]:
        floor = BASELINE_METRICS[column] - 0.02
        if metrics[column] < floor:
            failures.append(
                f"{column} {metrics[column]:.6f} < {floor:.6f}"
            )
    t4_improved = (
        metrics["evidence_quality"]
        > BASELINE_METRICS["evidence_quality"] + 1e-4
    )
    t4_same_with_lower_fpr = (
        metrics["evidence_quality"]
        >= BASELINE_METRICS["evidence_quality"] - 1e-4
        and misleading_config["real_false_positive_rate"]
        < BASELINE_METRICS["misleading_false_positive_rate"]
    )
    if not (t4_improved or t4_same_with_lower_fpr):
        failures.append(
            "T4 did not improve and did not match baseline with lower FPR"
        )
    if failures:
        raise RuntimeError("Quality gate failed: " + "; ".join(failures))
    print("Quality gate passed.")
    return metrics


quality_metrics = run_quality_gate(
    oof_true_df,
    oof_prediction_df,
    thresholds["misleading"],
)
'''


TRAIN_FULL = r'''# ==========================================
# 7. Full-data three-seed ensemble
# ==========================================

FULL_DATA_EPOCHS = max(1, int(np.median(best_epochs)))
print(f"Full-data epochs selected from fold median: {FULL_DATA_EPOCHS}")

full_train_df = pd.concat(
    [real_df, synthetic_df],
    ignore_index=True,
).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
full_class_weights = compute_fold_class_weights(full_train_df)

for seed in FULL_DATA_SEEDS:
    print(f"\n{'=' * 64}\nTraining full-data seed {seed}\n{'=' * 64}")
    set_seed(seed)
    full_loader = make_loader(
        full_train_df,
        tokenizer,
        shuffle=True,
        seed=seed,
    )
    model = ESGLoraMTLModel(MODEL_NAME).to(DEVICE)
    history, _, _ = train_epochs(
        model,
        full_loader,
        full_class_weights,
        FULL_DATA_EPOCHS,
    )
    member_dir = OUTPUT_DIR / f"full_seed_{seed}"
    member_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(member_dir / "training_history.csv", index=False)
    save_v3_model(
        model,
        member_dir,
        {
            "artifact_version": 3,
            "member_type": "full_data",
            "seed": seed,
            "epochs": FULL_DATA_EPOCHS,
            "real_rows": len(real_df),
            "synthetic_rows": len(synthetic_df),
        },
    )
    assert_save_load_parity(model, member_dir, full_loader)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
'''


TRAIN_SAVE = r'''# ==========================================
# 8. Save v3 inference config and optionally upload
# ==========================================

def save_inference_config():
    config = {
        "artifact_version": 3,
        "model_type": "ckip_bert_lora_multi_task_mlp",
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "head_ratio": HEAD_RATIO,
        "task_classes": TASK_CLASSES,
        "target_columns": TARGET_COLUMNS,
        "task_weights": TASK_WEIGHTS,
        "ensemble_members": [
            f"full_seed_{seed}" for seed in FULL_DATA_SEEDS
        ],
        "probability_ensemble": True,
        "temperatures": temperatures,
        "thresholds": thresholds,
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "use_rslora": True,
            "target_modules": "all-linear",
        },
        "pooling": "cls_masked_mean_masked_max",
        "official_timeline_label": "more_than_5_years",
        "quality_metrics": quality_metrics,
        "synthetic_policy": {
            "id_min": SYNTHETIC_ID_MIN,
            "t4_sample_weight": SYNTHETIC_T4_SAMPLE_WEIGHT,
            "source_group_holdout": True,
            "paired_rows": int(synthetic_df["pair_valid"].sum()),
            "total_rows": len(synthetic_df),
        },
        "artifact_layout": {
            "adapter": "{ensemble_member}/adapter/",
            "heads": "{ensemble_member}/heads.pt",
            "tokenizer": "tokenizer/",
            "calibration": "mtl_calibration.json",
            "thresholds": "mtl_thresholds.json",
        },
    }
    with open(INFERENCE_CONFIG_JSON, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    return config


def validate_artifacts():
    missing = []
    for seed in FULL_DATA_SEEDS:
        member_dir = OUTPUT_DIR / f"full_seed_{seed}"
        adapter_dir = member_dir / "adapter"
        for path in [
            adapter_dir / "adapter_config.json",
            member_dir / "heads.pt",
            member_dir / "metadata.json",
        ]:
            if not path.exists():
                missing.append(str(path))
        adapter_weights = [
            adapter_dir / "adapter_model.safetensors",
            adapter_dir / "adapter_model.bin",
        ]
        if not any(path.exists() for path in adapter_weights):
            missing.append(
                f"{adapter_dir}/adapter_model.safetensors|adapter_model.bin"
            )
    for path in [
        TOKENIZER_DIR,
        CALIBRATION_JSON,
        THRESHOLD_JSON,
        INFERENCE_CONFIG_JSON,
    ]:
        if not Path(path).exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing MTL v3 artifacts: " + ", ".join(missing))


def push_artifacts_to_hf(repo_id=HF_MTL_REPO_ID, private=HF_PRIVATE_REPO):
    save_inference_config()
    validate_artifacts()
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(OUTPUT_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message=HF_COMMIT_MESSAGE,
    )
    print(f"Uploaded CKIP-BERT LoRA artifacts: https://huggingface.co/{repo_id}")


inference_config = save_inference_config()
print(json.dumps(inference_config, ensure_ascii=False, indent=2))
validate_artifacts()

if RUN_HF_UPLOAD:
    notebook_login()
    push_artifacts_to_hf()
else:
    print("RUN_HF_UPLOAD is False. Artifacts remain local.")
'''


INFERENCE_MAIN = r'''import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


# ==========================================
# CKIP-BERT artifact v1/v2/v3 inference
# ==========================================

DEFAULT_MODEL_NAME = "ckiplab/bert-base-chinese"
DEFAULT_FOLDS = [1, 2, 3, 4, 5]
DEFAULT_MAX_LEN = 512
DEFAULT_HEAD_RATIO = 0.25
DEFAULT_T1_THRESHOLD = 0.5
DEFAULT_T3_THRESHOLD = 0.5
BATCH_SIZE = 16

ID_COLUMN = "id"
TEXT_COLUMN = "data"
TARGET_COLUMNS = [
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]
V3_TASK_CLASSES = {
    "t1": ["No", "Yes"],
    "t2": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years"],
    "t3": ["No", "Yes"],
    "t4": ["Clear", "Not Clear", "Misleading"],
}
LEGACY_TASK_CLASSES = {
    **V3_TASK_CLASSES,
    "t2": ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years"],
}
OFFICIAL_ALLOWED_VALUES = {
    "promise_status": {"No", "Yes"},
    "verification_timeline": {
        "N/A",
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "more_than_5_years",
    },
    "evidence_status": {"N/A", "No", "Yes"},
    "evidence_quality": {"N/A", "Clear", "Not Clear", "Misleading"},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def tokenize_head_tail(text, tokenizer, max_len, head_ratio):
    body_ids = tokenizer.encode(
        f"文本：{str(text)}",
        add_special_tokens=False,
        verbose=False,
    )
    max_body_len = max_len - 2
    if len(body_ids) > max_body_len:
        head_len = int(max_body_len * head_ratio)
        tail_len = max_body_len - head_len
        body_ids = body_ids[:head_len] + body_ids[-tail_len:]
    input_ids = [tokenizer.cls_token_id] + body_ids + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    pad_len = max_len - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len
    return input_ids, attention_mask


class ESGInferenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, head_ratio):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_ratio = head_ratio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        input_ids, attention_mask = tokenize_head_tail(
            self.df.iloc[index][TEXT_COLUMN],
            self.tokenizer,
            self.max_len,
            self.head_ratio,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class ESGLoraMTLModel(nn.Module):
    def __init__(self, model_name, adapter_dir):
        super().__init__()
        base_model = AutoModel.from_pretrained(model_name)
        self.backbone = PeftModel.from_pretrained(
            base_model,
            adapter_dir,
            is_trainable=False,
        )
        hidden_size = base_model.config.hidden_size
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.multi_sample_dropouts = nn.ModuleList(
            [nn.Dropout(probability) for probability in [0.10, 0.20, 0.30, 0.40, 0.50]]
        )
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(256, 128),
                    nn.LayerNorm(128),
                    nn.GELU(),
                    nn.Dropout(0.20),
                    nn.Linear(128, len(labels)),
                )
                for task, labels in V3_TASK_CLASSES.items()
            }
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        token_mask = attention_mask.unsqueeze(-1).bool()
        cls_pool = hidden[:, 0, :]
        mean_pool = (hidden * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        max_pool = hidden.masked_fill(~token_mask, torch.finfo(hidden.dtype).min).max(dim=1).values
        features = self.shared_mlp(
            torch.cat([cls_pool, mean_pool, max_pool], dim=-1)
        )
        return {
            task: torch.stack(
                [head(dropout(features)) for dropout in self.multi_sample_dropouts],
                dim=0,
            ).mean(dim=0)
            for task, head in self.heads.items()
        }

    def load_heads(self, path):
        state = torch.load(path, map_location=DEVICE, weights_only=True)
        self.shared_mlp.load_state_dict(state["shared_mlp"])
        self.heads.load_state_dict(state["heads"])


class LegacyESGUnifiedMTLModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.multi_sample_dropouts = nn.ModuleList(
            [nn.Dropout(probability) for probability in [0.1, 0.2, 0.3, 0.4, 0.5]]
        )
        self.t1_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
        self.t3_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
        self.t2_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4),
        )
        self.t4_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        t1_logits = torch.stack(
            [self.t1_head(dropout(cls_output)).squeeze(-1) for dropout in self.multi_sample_dropouts]
        ).mean(dim=0)
        t3_logits = torch.stack(
            [self.t3_head(dropout(cls_output)).squeeze(-1) for dropout in self.multi_sample_dropouts]
        ).mean(dim=0)
        return (
            t1_logits,
            self.t2_head(cls_output),
            t3_logits,
            self.t4_head(cls_output),
        )


def softmax_numpy(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def sigmoid_numpy(values):
    values = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-values))


def find_artifact_root(path):
    path = Path(path)
    for candidate in [path, path / "mtl_outputs"]:
        if (candidate / "mtl_inference_config.json").exists():
            return candidate
    return None


def resolve_artifact_root(repo_id=None, model_dir=None):
    if repo_id:
        config_filename = "mtl_inference_config.json"
        try:
            config_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=config_filename,
                )
            )
        except Exception:
            config_filename = "mtl_outputs/mtl_inference_config.json"
            config_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=config_filename,
                )
            )
        with open(config_path, "r", encoding="utf-8") as file:
            remote_config = json.load(file)
        version = int(remote_config.get("artifact_version", 1))
        if version == 3:
            allow_patterns = [
                config_filename,
                "mtl_thresholds.json",
                "mtl_calibration.json",
                "tokenizer/**",
                "full_seed_*/adapter/**",
                "full_seed_*/heads.pt",
                "full_seed_*/metadata.json",
                "mtl_outputs/mtl_thresholds.json",
                "mtl_outputs/mtl_calibration.json",
                "mtl_outputs/tokenizer/**",
                "mtl_outputs/full_seed_*/adapter/**",
                "mtl_outputs/full_seed_*/heads.pt",
                "mtl_outputs/full_seed_*/metadata.json",
            ]
        else:
            allow_patterns = [
                config_filename,
                "mtl_thresholds.json",
                "tokenizer/**",
                "fold_*/best_model.pth",
                "best_mtl_model_fold_*.pth",
                "mtl_outputs/mtl_thresholds.json",
                "mtl_outputs/tokenizer/**",
                "mtl_outputs/fold_*/best_model.pth",
                "mtl_outputs/best_mtl_model_fold_*.pth",
            ]
        downloaded = snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
        )
        root = find_artifact_root(downloaded)
        return root if root is not None else Path(downloaded)
    model_dir = Path("mtl_outputs") if model_dir is None else Path(model_dir)
    root = find_artifact_root(model_dir)
    if root is None:
        raise FileNotFoundError(
            f"Could not find CKIP MTL artifacts under {model_dir}."
        )
    return root


def load_config(root):
    path = Path(root) / "mtl_inference_config.json"
    if not path.exists():
        return {
            "artifact_version": 1,
            "model_name": DEFAULT_MODEL_NAME,
            "folds": DEFAULT_FOLDS,
            "max_len": DEFAULT_MAX_LEN,
            "head_ratio": DEFAULT_HEAD_RATIO,
            "thresholds": {
                "t1_threshold": DEFAULT_T1_THRESHOLD,
                "t3_threshold": DEFAULT_T3_THRESHOLD,
            },
        }
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)
    if int(config.get("artifact_version", 1)) not in {1, 2, 3}:
        raise ValueError(
            f"Unsupported artifact version: {config.get('artifact_version')}"
        )
    return config


def make_data_loader(test_df, tokenizer, max_len, head_ratio):
    return DataLoader(
        ESGInferenceDataset(
            test_df,
            tokenizer,
            max_len,
            head_ratio,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=USE_AMP,
    )


def predict_v3(model, data_loader, temperatures):
    output = {task: [] for task in V3_TASK_CLASSES}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                logits = model(input_ids, attention_mask)
            for task in V3_TASK_CLASSES:
                values = logits[task].float().cpu().numpy()
                output[task].append(
                    softmax_numpy(
                        values / float(temperatures.get(task, 1.0))
                    )
                )
    return {
        task: np.concatenate(parts, axis=0)
        for task, parts in output.items()
    }


def predict_legacy(model, data_loader):
    output = {task: [] for task in LEGACY_TASK_CLASSES}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                logits = model(input_ids, attention_mask)
            output["t1"].append(
                np.column_stack(
                    [
                        1.0 - sigmoid_numpy(logits[0].float().cpu().numpy()),
                        sigmoid_numpy(logits[0].float().cpu().numpy()),
                    ]
                )
            )
            output["t2"].append(
                softmax_numpy(logits[1].float().cpu().numpy())
            )
            output["t3"].append(
                np.column_stack(
                    [
                        1.0 - sigmoid_numpy(logits[2].float().cpu().numpy()),
                        sigmoid_numpy(logits[2].float().cpu().numpy()),
                    ]
                )
            )
            output["t4"].append(
                softmax_numpy(logits[3].float().cpu().numpy())
            )
    return {
        task: np.concatenate(parts, axis=0)
        for task, parts in output.items()
    }


def choose_t4_label(probabilities, labels, misleading_config):
    if misleading_config.get("legacy_argmax", False):
        return labels[int(np.argmax(probabilities))]
    clear_index = labels.index("Clear")
    not_clear_index = labels.index("Not Clear")
    misleading_index = labels.index("Misleading")
    standard_index = (
        clear_index
        if probabilities[clear_index] >= probabilities[not_clear_index]
        else not_clear_index
    )
    probability_threshold = float(
        misleading_config.get("probability_threshold", 1.0)
    )
    margin_threshold = float(
        misleading_config.get("margin_threshold", 1.0)
    )
    if (
        probabilities[misleading_index] >= probability_threshold
        and probabilities[misleading_index] - probabilities[standard_index]
        >= margin_threshold
    ):
        return "Misleading"
    return labels[standard_index]


def route_predictions(
    test_df,
    probabilities,
    task_classes,
    t1_threshold,
    t3_threshold,
    misleading_config=None,
):
    misleading_config = misleading_config or {
        "probability_threshold": 0.0,
        "margin_threshold": 0.0,
    }
    results = []
    for index in range(len(test_df)):
        if probabilities["t1"][index, 1] < t1_threshold:
            results.append(
                {
                    ID_COLUMN: test_df.iloc[index][ID_COLUMN],
                    "promise_status": "No",
                    "verification_timeline": "N/A",
                    "evidence_status": "N/A",
                    "evidence_quality": "N/A",
                }
            )
            continue
        t2_label = task_classes["t2"][
            int(probabilities["t2"][index].argmax())
        ]
        if t2_label == "longer_than_5_years":
            t2_label = "more_than_5_years"
        if probabilities["t3"][index, 1] < t3_threshold:
            results.append(
                {
                    ID_COLUMN: test_df.iloc[index][ID_COLUMN],
                    "promise_status": "Yes",
                    "verification_timeline": t2_label,
                    "evidence_status": "No",
                    "evidence_quality": "N/A",
                }
            )
            continue
        results.append(
            {
                ID_COLUMN: test_df.iloc[index][ID_COLUMN],
                "promise_status": "Yes",
                "verification_timeline": t2_label,
                "evidence_status": "Yes",
                "evidence_quality": choose_t4_label(
                    probabilities["t4"][index],
                    task_classes["t4"],
                    misleading_config,
                ),
            }
        )
    return pd.DataFrame(results)[[ID_COLUMN] + TARGET_COLUMNS]


def validate_input(df):
    missing = [
        column
        for column in [ID_COLUMN, TEXT_COLUMN]
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"Test CSV missing columns: {missing}")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Test CSV contains duplicated ids.")


def validate_output(output_df, test_df):
    expected_columns = [ID_COLUMN] + TARGET_COLUMNS
    if list(output_df.columns) != expected_columns:
        raise ValueError(
            f"Unexpected output columns: {list(output_df.columns)}"
        )
    if len(output_df) != len(test_df):
        raise ValueError("Output row count differs from test input.")
    if output_df[ID_COLUMN].tolist() != test_df[ID_COLUMN].tolist():
        raise ValueError("Output id order differs from test input.")
    for column, allowed in OFFICIAL_ALLOWED_VALUES.items():
        values = set(output_df[column].fillna("N/A").astype(str))
        invalid = values - allowed
        if invalid:
            raise ValueError(
                f"Invalid values in {column}: {sorted(invalid)}"
            )
    no_promise = output_df["promise_status"].eq("No")
    if not (
        output_df.loc[
            no_promise,
            ["verification_timeline", "evidence_status", "evidence_quality"],
        ]
        .fillna("N/A")
        .eq("N/A")
        .all(axis=None)
    ):
        raise ValueError("promise_status=No routing is invalid.")
    no_evidence = (
        output_df["promise_status"].eq("Yes")
        & output_df["evidence_status"].eq("No")
    )
    if not output_df.loc[
        no_evidence,
        "evidence_quality",
    ].fillna("N/A").eq("N/A").all():
        raise ValueError("evidence_status=No routing is invalid.")


def resolve_legacy_checkpoint(root, fold, repo_id=None):
    root = Path(root)
    new_path = root / f"fold_{fold}" / "best_model.pth"
    if new_path.exists():
        return new_path
    legacy_path = root / f"best_mtl_model_fold_{fold}.pth"
    if legacy_path.exists():
        return legacy_path
    if repo_id:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=f"best_mtl_model_fold_{fold}.pth",
            )
        )
    raise FileNotFoundError(f"Missing legacy checkpoint for fold {fold}.")


def ensemble_v3(root, config, data_loader):
    members = config.get("ensemble_members", [])
    if not members:
        raise ValueError("Artifact v3 has no ensemble_members.")
    temperatures = config.get("temperatures", {})
    accumulated = {
        task: None
        for task in V3_TASK_CLASSES
    }
    for member in members:
        member_dir = Path(root) / member
        print(f"Loading CKIP-BERT LoRA member {member}")
        model = ESGLoraMTLModel(
            config.get("model_name", DEFAULT_MODEL_NAME),
            member_dir / "adapter",
        ).to(DEVICE)
        model.load_heads(member_dir / "heads.pt")
        member_probabilities = predict_v3(
            model,
            data_loader,
            temperatures,
        )
        for task, values in member_probabilities.items():
            if accumulated[task] is None:
                accumulated[task] = np.zeros_like(
                    values,
                    dtype=np.float64,
                )
            accumulated[task] += values
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        task: values / len(members)
        for task, values in accumulated.items()
    }


def ensemble_legacy(root, config, data_loader, repo_id=None):
    folds = [int(fold) for fold in config.get("folds", DEFAULT_FOLDS)]
    accumulated = {task: None for task in LEGACY_TASK_CLASSES}
    for fold in folds:
        print(f"Loading legacy CKIP-BERT fold {fold}")
        checkpoint_path = resolve_legacy_checkpoint(
            root,
            fold,
            repo_id=repo_id,
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model = LegacyESGUnifiedMTLModel(
            config.get("model_name", DEFAULT_MODEL_NAME)
        ).to(DEVICE)
        model.load_state_dict(state_dict)
        fold_probabilities = predict_legacy(model, data_loader)
        for task, values in fold_probabilities.items():
            if accumulated[task] is None:
                accumulated[task] = np.zeros_like(
                    values,
                    dtype=np.float64,
                )
            accumulated[task] += values
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        task: values / len(folds)
        for task, values in accumulated.items()
    }


def ensemble_inference_and_export(
    repo_id,
    test_csv_path,
    output_csv_path="final_submission.csv",
    model_dir=None,
):
    root = resolve_artifact_root(repo_id=repo_id, model_dir=model_dir)
    config = load_config(root)
    version = int(config.get("artifact_version", 1))
    model_name = config.get("model_name", DEFAULT_MODEL_NAME)
    max_len = int(config.get("max_len", DEFAULT_MAX_LEN))
    head_ratio = float(config.get("head_ratio", DEFAULT_HEAD_RATIO))
    thresholds = config.get("thresholds", {})
    t1_threshold = float(
        thresholds.get("t1_threshold", DEFAULT_T1_THRESHOLD)
    )
    t3_threshold = float(
        thresholds.get("t3_threshold", DEFAULT_T3_THRESHOLD)
    )

    test_df = pd.read_csv(test_csv_path).reset_index(drop=True)
    validate_input(test_df)
    tokenizer_path = Path(root) / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path) if tokenizer_path.exists() else model_name,
        use_fast=True,
    )
    data_loader = make_data_loader(
        test_df,
        tokenizer,
        max_len,
        head_ratio,
    )

    if version == 3:
        probabilities = ensemble_v3(root, config, data_loader)
        task_classes = V3_TASK_CLASSES
        misleading_config = thresholds.get("misleading", {})
    else:
        probabilities = ensemble_legacy(
            root,
            config,
            data_loader,
            repo_id=repo_id,
        )
        task_classes = LEGACY_TASK_CLASSES
        misleading_config = {
            "legacy_argmax": True,
        }

    output_df = route_predictions(
        test_df,
        probabilities,
        task_classes,
        t1_threshold=t1_threshold,
        t3_threshold=t3_threshold,
        misleading_config=misleading_config,
    )
    validate_output(output_df, test_df)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Exported CKIP-BERT submission to {output_csv_path}")
    print(output_df.head())
    return output_df
'''


INFERENCE_RUN = r'''# ==========================================
# Run inference
# ==========================================

# Local v3 or legacy artifacts:
# ensemble_inference_and_export(
#     repo_id=None,
#     test_csv_path="/content/test.csv",
#     output_csv_path="final_submission.csv",
#     model_dir="/content/mtl_outputs",
# )

# Hugging Face artifacts:
DEFAULT_REPO_ID = "maxbeettww/VeriPromise_ESG_2026_9906"
TEST_CSV_PATH = "../data/ori_data/vpesg4k_test_2000.csv"

ensemble_inference_and_export(
    repo_id=DEFAULT_REPO_ID,
    test_csv_path=TEST_CSV_PATH,
    output_csv_path="final_submission.csv",
)
'''


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "veripromiseesg-2026-team-9906",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_train_notebook() -> dict:
    return notebook(
        [
            markdown(
                '<a href="https://colab.research.google.com/github/Heng1222/'
                'VeriPromiseESG_2026_TEAM_9906/blob/feat-model-train/app/model/'
                'model_train.ipynb" target="_parent"><img '
                'src="https://colab.research.google.com/assets/colab-badge.svg" '
                'alt="Open In Colab"/></a>\n'
            ),
            markdown(
                "# CKIP-BERT LoRA Multi-Task Training\n\n"
                "Artifact v3 uses RSLoRA across all CKIP-BERT linear layers, "
                "class-balanced focal objectives, grouped synthetic holdouts, "
                "OOF temperature calibration, stable routing thresholds, and "
                "three full-data ensemble members. External CSV input/output "
                "contracts remain unchanged.\n"
            ),
            code(
                "# Colab dependency installation. Restart the runtime if requested.\n"
                "# !pip install -q transformers peft accelerate torch pandas numpy "
                "scikit-learn tqdm huggingface_hub safetensors\n"
            ),
            code(TRAIN_IMPORTS_CONFIG),
            code(TRAIN_DATA),
            code(TRAIN_MODEL),
            code(TRAIN_METRICS),
            code(TRAIN_ARTIFACTS),
            code(TRAIN_LOOPS),
            code(TRAIN_CALIBRATE),
            code(TRAIN_FULL),
            code(TRAIN_SAVE),
        ]
    )


def build_inference_notebook() -> dict:
    return notebook(
        [
            markdown(
                '<a href="https://colab.research.google.com/github/Heng1222/'
                'VeriPromiseESG_2026_TEAM_9906/blob/feat-model-train/app/model/'
                'model_inference.ipynb" target="_parent"><img '
                'src="https://colab.research.google.com/assets/colab-badge.svg" '
                'alt="Open In Colab"/></a>\n'
            ),
            markdown(
                "# CKIP-BERT Multi-Task Inference\n\n"
                "Loads artifact v3 LoRA full-data ensembles and remains backward "
                "compatible with v1/v2 full checkpoints. Input is `id,data`; "
                "output is the official five-column submission schema.\n"
            ),
            code(
                "# Colab dependency installation. Restart the runtime if requested.\n"
                "# !pip install -q transformers peft accelerate torch pandas numpy "
                "tqdm huggingface_hub safetensors\n"
            ),
            code(INFERENCE_MAIN),
            code(INFERENCE_RUN),
        ]
    )


def write_notebooks() -> None:
    outputs = {
        ROOT / "model_train.ipynb": build_train_notebook(),
        ROOT / "model_inference.ipynb": build_inference_notebook(),
    }
    for path, content in outputs.items():
        path.write_text(
            json.dumps(content, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    write_notebooks()
