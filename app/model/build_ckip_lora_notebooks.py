"""Generate the simplified CKIP-BERT MLM + LoRA notebooks."""

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


TRAIN_CONFIG = r'''import gc
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
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)

warnings.filterwarnings("ignore")


# ==========================================
# 0. Configuration
# ==========================================

MODEL_NAME = "ckiplab/bert-base-chinese"
SEED = 42
FOLDS = [1, 2, 3, 4, 5]

MLM_MAX_LEN = 512
MLM_STRIDE = 64
MLM_PROBABILITY = 0.15
MLM_BATCH_SIZE = 4
MLM_GRAD_ACCUM_STEPS = 4
MLM_EPOCHS = 6
MLM_LR = 3e-5

MAX_LEN = 512
HEAD_RATIO = 0.25
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
MIN_F1_IMPROVEMENT = 1e-4
LORA_LEARNING_RATE = 5e-5
HEAD_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.08
MAX_GRAD_NORM = 1.0
CLASS_WEIGHT_POWER = 0.5
MAX_CLASS_WEIGHT = 5.0

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.10

T1_THRESHOLD = 0.5
T3_THRESHOLD = 0.5

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
    "t2": [
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "more_than_5_years",
    ],
    "t3": ["No", "Yes"],
    "t4": ["Clear", "Not Clear", "Misleading"],
}
TASK_LABEL_MAPS = {
    task: {label: index for index, label in enumerate(labels)}
    for task, labels in TASK_CLASSES.items()
}
COMPETITION_SCORE_WEIGHTS = {
    "promise_status": 0.20,
    "verification_timeline": 0.15,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
}

MLM_CORPUS_FILES = {
    "train": (Path("ori_data") / "vpesg4k_train_1000 V1.csv", 1000),
    "val": (Path("ori_data") / "vpesg4k_val_1000.csv", 1000),
    "test": (Path("ori_data") / "vpesg4k_test_2000.csv", 2000),
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

PROJECT_ROOT = Path.cwd().resolve()
for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (candidate / "app" / "data" / "ori_data").exists():
        PROJECT_ROOT = candidate
        break

LOCAL_DATA_DIR = PROJECT_ROOT / "app" / "data"
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/Heng1222/"
    "VeriPromiseESG_2026_TEAM_9906/feat-model-train/app/data/"
)

OUTPUT_DIR = Path("mtl_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MLM_BACKBONE_DIR = OUTPUT_DIR / "mlm_backbone"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
MLM_CONFIG_JSON = OUTPUT_DIR / "mlm_config.json"
MLM_HISTORY_CSV = OUTPUT_DIR / "mlm_training_history.csv"
OOF_PREDICTIONS_CSV = OUTPUT_DIR / "oof_predictions.csv"
TRAINING_HISTORY_CSV = OUTPUT_DIR / "training_history.csv"
INFERENCE_CONFIG_JSON = OUTPUT_DIR / "mtl_inference_config.json"

RUN_HF_UPLOAD = False
HF_REPO_ID = "maxbeettww/VeriPromise_ESG_2026_9906"
HF_PRIVATE_REPO = False

print(f"Device: {DEVICE}")
print(f"Project root: {PROJECT_ROOT}")
'''


TRAIN_DATA = r'''# ==========================================
# 1. Data loading and tokenization
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    relative_path = Path(relative_path)
    local_path = LOCAL_DATA_DIR / relative_path
    if local_path.exists():
        return pd.read_csv(local_path)
    remote_path = relative_path.as_posix().replace(" ", "%20")
    return pd.read_csv(f"{RAW_BASE_URL}{remote_path}")


def load_mlm_corpus():
    parts = []
    counts = {}
    for split, (relative_path, expected_rows) in MLM_CORPUS_FILES.items():
        frame = read_csv_local_or_remote(relative_path)
        if TEXT_COLUMN not in frame.columns:
            raise ValueError(f"{relative_path} missing {TEXT_COLUMN}.")
        text = frame[[TEXT_COLUMN]].copy()
        invalid = (
            text[TEXT_COLUMN].isna()
            | text[TEXT_COLUMN].astype(str).str.strip().eq("")
        )
        if invalid.any():
            raise ValueError(f"{relative_path} contains blank data rows.")
        if len(text) != expected_rows:
            raise ValueError(
                f"{relative_path} has {len(text)} rows; expected {expected_rows}."
            )
        text[TEXT_COLUMN] = text[TEXT_COLUMN].astype(str)
        parts.append(text)
        counts[split] = len(text)
    corpus = pd.concat(parts, ignore_index=True)
    if len(corpus) != 4000:
        raise ValueError(f"MLM corpus has {len(corpus)} rows; expected 4000.")
    print(f"MLM corpus: {counts}, total={len(corpus)}")
    return corpus


def load_fold_data(fold):
    if fold not in FOLDS:
        raise ValueError(f"Unknown fold: {fold}")
    required = [ID_COLUMN, TEXT_COLUMN] + TARGET_COLUMNS
    output = {}
    for split, expected_rows in [("train", 1711), ("val", 400)]:
        relative_path = (
            Path("clean_data") / f"{split}_fold_{fold}.csv"
        )
        frame = read_csv_local_or_remote(relative_path).copy()
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{relative_path} missing columns: {missing}")
        if len(frame) != expected_rows:
            raise ValueError(
                f"{relative_path} has {len(frame)} rows; expected {expected_rows}."
            )
        if frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"{relative_path} contains duplicated ids.")
        for column in TARGET_COLUMNS:
            frame[column] = frame[column].apply(normalize_value)
        output[split] = frame
    print(
        f"Fold {fold}: train={len(output['train'])}, "
        f"val={len(output['val'])}, "
        f"synthetic_train="
        f"{int((pd.to_numeric(output['train'][ID_COLUMN]) >= 90000).sum())}"
    )
    return output["train"], output["val"]


def tokenize_head_tail(text, tokenizer, max_len=MAX_LEN):
    body_ids = tokenizer.encode(
        str(text),
        add_special_tokens=False,
        verbose=False,
    )
    max_body_len = max_len - 2
    if len(body_ids) > max_body_len:
        head_len = int(max_body_len * HEAD_RATIO)
        tail_len = max_body_len - head_len
        body_ids = body_ids[:head_len] + body_ids[-tail_len:]
    input_ids = [tokenizer.cls_token_id] + body_ids + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    pad_len = max_len - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len
    return input_ids, attention_mask


class ClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, include_labels=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.include_labels = include_labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        input_ids, attention_mask = tokenize_head_tail(
            row[TEXT_COLUMN],
            self.tokenizer,
        )
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        if self.include_labels:
            for task, column in TASK_COLUMNS.items():
                value = normalize_value(row.get(column))
                item[f"{task}_label"] = torch.tensor(
                    TASK_LABEL_MAPS[task].get(value, -100),
                    dtype=torch.long,
                )
        return item


def make_loader(
    dataframe,
    tokenizer,
    shuffle=False,
    include_labels=True,
    seed=SEED,
):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ClassificationDataset(
            dataframe,
            tokenizer,
            include_labels=include_labels,
        ),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=USE_AMP,
    )
'''


TRAIN_MLM = r'''# ==========================================
# 2. ESG masked language modeling
# ==========================================

class MLMDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def build_mlm_examples(corpus, tokenizer):
    examples = []
    for text in tqdm(corpus[TEXT_COLUMN], desc="Tokenizing MLM corpus"):
        encoded = tokenizer(
            str(text),
            add_special_tokens=True,
            truncation=True,
            max_length=MLM_MAX_LEN,
            stride=MLM_STRIDE,
            return_overflowing_tokens=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )
        for input_ids, attention_mask, special_tokens_mask in zip(
            encoded["input_ids"],
            encoded["attention_mask"],
            encoded["special_tokens_mask"],
        ):
            examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "special_tokens_mask": special_tokens_mask,
                }
            )
    if not examples:
        raise ValueError("MLM tokenization produced no examples.")
    return examples


def train_mlm_backbone(corpus, tokenizer):
    examples = build_mlm_examples(corpus, tokenizer)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
        pad_to_multiple_of=8 if USE_AMP else None,
    )
    loader = DataLoader(
        MLMDataset(examples),
        batch_size=MLM_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=collator,
        num_workers=0,
        pin_memory=USE_AMP,
    )
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MLM_LR,
        weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = math.ceil(len(loader) / MLM_GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * MLM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * WARMUP_RATIO)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    history = []

    for epoch in range(1, MLM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, batch in enumerate(
            tqdm(loader, desc=f"MLM {epoch}/{MLM_EPOCHS}"),
            start=1,
        ):
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                loss = model(**batch).loss
            scaler.scale(loss / MLM_GRAD_ACCUM_STEPS).backward()
            if step % MLM_GRAD_ACCUM_STEPS == 0 or step == len(loader):
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

        train_loss = loss_sum / len(loader)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "perplexity": math.exp(min(train_loss, 20.0)),
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(pd.DataFrame([row]).to_string(index=False))

    if MLM_BACKBONE_DIR.exists():
        shutil.rmtree(MLM_BACKBONE_DIR)
    model.base_model.save_pretrained(
        MLM_BACKBONE_DIR,
        safe_serialization=True,
    )
    history_df = pd.DataFrame(history)
    history_df.to_csv(MLM_HISTORY_CSV, index=False)
    mlm_config = {
        "model_name": MODEL_NAME,
        "documents": len(corpus),
        "training_examples": len(examples),
        "max_length": MLM_MAX_LEN,
        "stride": MLM_STRIDE,
        "mask_probability": MLM_PROBABILITY,
        "epochs": MLM_EPOCHS,
        "batch_size": MLM_BATCH_SIZE,
        "gradient_accumulation_steps": MLM_GRAD_ACCUM_STEPS,
        "learning_rate": MLM_LR,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "corpus": {
            split: {
                "path": path.as_posix(),
                "rows": rows,
            }
            for split, (path, rows) in MLM_CORPUS_FILES.items()
        },
    }
    with open(MLM_CONFIG_JSON, "w", encoding="utf-8") as file:
        json.dump(mlm_config, file, ensure_ascii=False, indent=2)

    del model, loader, examples
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history_df, mlm_config
'''


TRAIN_CLASSIFIER = r'''# ==========================================
# 3. LoRA + shared MLP + task heads
# ==========================================

def make_lora_config():
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        bias="none",
    )


class SimpleESGModel(nn.Module):
    def __init__(
        self,
        backbone_path,
        adapter_dir=None,
        is_trainable=True,
    ):
        super().__init__()
        base_model = AutoModel.from_pretrained(backbone_path)
        if adapter_dir is None:
            self.backbone = get_peft_model(base_model, make_lora_config())
        else:
            self.backbone = PeftModel.from_pretrained(
                base_model,
                adapter_dir,
                is_trainable=is_trainable,
            )
        hidden_size = base_model.config.hidden_size
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Dropout(0.10),
                    nn.Linear(128, len(labels)),
                )
                for task, labels in TASK_CLASSES.items()
            }
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        features = self.shared_mlp(hidden[:, 0, :])
        return {
            task: head(features)
            for task, head in self.heads.items()
        }

    def head_state_dict(self):
        return {
            "shared_mlp": self.shared_mlp.state_dict(),
            "heads": self.heads.state_dict(),
        }

    def load_heads(self, state):
        self.shared_mlp.load_state_dict(state["shared_mlp"])
        self.heads.load_state_dict(state["heads"])


def compute_class_weights(dataframe):
    labels = {
        task: dataframe[column].map(TASK_LABEL_MAPS[task]).fillna(-100).astype(int)
        for task, column in TASK_COLUMNS.items()
    }
    masks = {
        "t1": labels["t1"] >= 0,
        "t2": (labels["t1"] == 1) & (labels["t2"] >= 0),
        "t3": (labels["t1"] == 1) & (labels["t3"] >= 0),
        "t4": (
            (labels["t1"] == 1)
            & (labels["t3"] == 1)
            & (labels["t4"] >= 0)
        ),
    }
    output = {}
    for task, valid in masks.items():
        counts = np.bincount(
            labels[task][valid].to_numpy(),
            minlength=len(TASK_CLASSES[task]),
        ).astype(np.float64)
        largest = max(float(counts.max()), 1.0)
        weights = np.power(
            largest / np.maximum(counts, 1.0),
            CLASS_WEIGHT_POWER,
        )
        weights = np.minimum(weights, MAX_CLASS_WEIGHT)
        output[task] = torch.tensor(
            weights,
            dtype=torch.float,
            device=DEVICE,
        )
        print(
            f"{task} counts={counts.astype(int).tolist()} "
            f"weights={weights.round(3).tolist()}"
        )
    return output


def calculate_loss(logits, batch, class_weights):
    labels = {
        task: batch[f"{task}_label"].to(DEVICE)
        for task in TASK_CLASSES
    }
    masks = {
        "t1": labels["t1"] >= 0,
        "t2": (labels["t1"] == 1) & (labels["t2"] >= 0),
        "t3": (labels["t1"] == 1) & (labels["t3"] >= 0),
        "t4": (
            (labels["t1"] == 1)
            & (labels["t3"] == 1)
            & (labels["t4"] >= 0)
        ),
    }
    task_losses = {}
    for task, valid in masks.items():
        if valid.any():
            task_losses[task] = F.cross_entropy(
                logits[task][valid],
                labels[task][valid],
                weight=class_weights[task],
            )
    if not task_losses:
        raise ValueError("Batch contains no valid classification labels.")
    return sum(task_losses.values()) / len(task_losses), task_losses


def predict_probabilities(model, loader):
    model.eval()
    output = {task: [] for task in TASK_CLASSES}
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                logits = model(input_ids, attention_mask)
            for task in TASK_CLASSES:
                output[task].append(
                    torch.softmax(logits[task], dim=-1).cpu().numpy()
                )
    return {
        task: np.concatenate(parts, axis=0)
        for task, parts in output.items()
    }


def route_predictions(dataframe, probabilities):
    rows = []
    for index, row in dataframe.reset_index(drop=True).iterrows():
        t1 = (
            "Yes"
            if probabilities["t1"][index, 1] >= T1_THRESHOLD
            else "No"
        )
        if t1 == "No":
            rows.append(
                {
                    ID_COLUMN: row[ID_COLUMN],
                    "promise_status": "No",
                    "verification_timeline": "N/A",
                    "evidence_status": "N/A",
                    "evidence_quality": "N/A",
                }
            )
            continue

        t2 = TASK_CLASSES["t2"][
            int(probabilities["t2"][index].argmax())
        ]
        t3 = (
            "Yes"
            if probabilities["t3"][index, 1] >= T3_THRESHOLD
            else "No"
        )
        if t3 == "No":
            rows.append(
                {
                    ID_COLUMN: row[ID_COLUMN],
                    "promise_status": "Yes",
                    "verification_timeline": t2,
                    "evidence_status": "No",
                    "evidence_quality": "N/A",
                }
            )
            continue

        t4 = TASK_CLASSES["t4"][
            int(probabilities["t4"][index].argmax())
        ]
        rows.append(
            {
                ID_COLUMN: row[ID_COLUMN],
                "promise_status": "Yes",
                "verification_timeline": t2,
                "evidence_status": "Yes",
                "evidence_quality": t4,
            }
        )
    return pd.DataFrame(rows)[[ID_COLUMN] + TARGET_COLUMNS]


def evaluate_predictions(true_df, prediction_df):
    merged = true_df[[ID_COLUMN] + TARGET_COLUMNS].merge(
        prediction_df,
        on=ID_COLUMN,
        suffixes=("_true", "_pred"),
        validate="one_to_one",
    )
    scores = {}
    for task, column in TASK_COLUMNS.items():
        true_values = merged[f"{column}_true"].apply(normalize_value)
        predicted_values = merged[f"{column}_pred"].apply(normalize_value)
        valid = true_values.notna()
        scores[column] = f1_score(
            true_values[valid],
            predicted_values[valid].fillna("N/A"),
            labels=TASK_CLASSES[task],
            average="macro",
            zero_division=0,
        )
    scores["mean_macro_f1"] = float(np.mean(list(scores.values())))
    scores["competition_macro_f1"] = float(
        sum(
            scores[column] * weight
            for column, weight in COMPETITION_SCORE_WEIGHTS.items()
        )
    )
    return scores


def evaluate_loss(model, loader, class_weights):
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                logits = model(input_ids, attention_mask)
                loss, _ = calculate_loss(logits, batch, class_weights)
            loss_sum += float(loss.cpu())
    return loss_sum / len(loader)


def save_classifier(model, output_dir, metadata):
    output_dir = Path(output_dir)
    adapter_dir = output_dir / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(adapter_dir, safe_serialization=True)
    torch.save(model.head_state_dict(), output_dir / "heads.pt")
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def load_classifier(output_dir, is_trainable=False):
    output_dir = Path(output_dir)
    model = SimpleESGModel(
        MLM_BACKBONE_DIR,
        adapter_dir=output_dir / "adapter",
        is_trainable=is_trainable,
    ).to(DEVICE)
    state = torch.load(
        output_dir / "heads.pt",
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_heads(state)
    return model


def train_classifier(fold, train_df, val_df, tokenizer):
    train_loader = make_loader(
        train_df,
        tokenizer,
        shuffle=True,
        seed=SEED + fold,
    )
    val_loader = make_loader(val_df, tokenizer)
    class_weights = compute_class_weights(train_df)
    model = SimpleESGModel(MLM_BACKBONE_DIR).to(DEVICE)
    model.backbone.print_trainable_parameters()
    lora_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    trainable_parameters = lora_parameters + head_parameters
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": LORA_LEARNING_RATE,
            },
            {
                "params": head_parameters,
                "lr": HEAD_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * MAX_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * WARMUP_RATIO)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    best_score = -1.0
    epochs_without_improvement = 0
    history = []
    fold_dir = OUTPUT_DIR / f"fold_{fold}"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Classifier {epoch}/{MAX_EPOCHS}"),
            start=1,
        ):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                logits = model(input_ids, attention_mask)
                loss, _ = calculate_loss(
                    logits,
                    batch,
                    class_weights,
                )
            scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    MAX_GRAD_NORM,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            train_loss_sum += float(loss.detach().cpu())

        val_loss = evaluate_loss(model, val_loader, class_weights)
        probabilities = predict_probabilities(model, val_loader)
        predictions = route_predictions(val_df, probabilities)
        metrics = evaluate_predictions(val_df, predictions)
        current_lrs = scheduler.get_last_lr()
        row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss_sum / len(train_loader),
            "val_loss": val_loss,
            "lora_learning_rate": current_lrs[0],
            "head_learning_rate": current_lrs[1],
            **metrics,
        }
        history.append(row)
        print(pd.DataFrame([row]).to_string(index=False))

        if metrics["competition_macro_f1"] > best_score + MIN_F1_IMPROVEMENT:
            best_score = metrics["competition_macro_f1"]
            epochs_without_improvement = 0
            save_classifier(
                model,
                fold_dir,
                {
                    "artifact_version": 8,
                    "fold": fold,
                    "best_epoch": epoch,
                    "best_val_loss": val_loss,
                    "best_competition_macro_f1": best_score,
                    "mean_macro_f1_at_best_epoch": metrics["mean_macro_f1"],
                    "max_epochs": MAX_EPOCHS,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "min_f1_improvement": MIN_F1_IMPROVEMENT,
                    "lora_learning_rate": LORA_LEARNING_RATE,
                    "head_learning_rate": HEAD_LEARNING_RATE,
                    "warmup_ratio": WARMUP_RATIO,
                    "weight_decay": WEIGHT_DECAY,
                    "class_weights": {
                        task: weights.detach().cpu().tolist()
                        for task, weights in class_weights.items()
                    },
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after epoch {epoch}.")
                break

    history_df = pd.DataFrame(history)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_model = load_classifier(fold_dir)
    probabilities = predict_probabilities(best_model, val_loader)
    predictions = route_predictions(val_df, probabilities)
    metrics = evaluate_predictions(val_df, predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    for task, column in TASK_COLUMNS.items():
        true_values = val_df[column].apply(normalize_value)
        predicted_values = predictions[column].apply(normalize_value)
        valid = true_values.notna()
        print(f"\n=== {column} ===")
        print(
            classification_report(
                true_values[valid],
                predicted_values[valid].fillna("N/A"),
                labels=TASK_CLASSES[task],
                zero_division=0,
            )
        )
    predictions["fold"] = fold
    return best_model, history_df, metrics, predictions
'''


TRAIN_RUN = r'''# ==========================================
# 4. Run MLM, then train five classifiers
# ==========================================

set_seed(SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.save_pretrained(TOKENIZER_DIR)

mlm_corpus = load_mlm_corpus()
mlm_history, mlm_config = train_mlm_backbone(mlm_corpus, tokenizer)
del mlm_corpus
gc.collect()

history_parts = []
oof_true_parts = []
oof_prediction_parts = []
fold_metrics = {}

for fold in FOLDS:
    print(f"\n{'=' * 64}\nTraining fold {fold}\n{'=' * 64}")
    set_seed(SEED + fold)
    train_df, val_df = load_fold_data(fold)
    model, fold_history, metrics, predictions = train_classifier(
        fold,
        train_df,
        val_df,
        tokenizer,
    )
    history_parts.append(fold_history)
    oof_true_parts.append(val_df[[ID_COLUMN] + TARGET_COLUMNS].copy())
    oof_prediction_parts.append(predictions)
    fold_metrics[str(fold)] = metrics
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

training_history = pd.concat(history_parts, ignore_index=True)
training_history.to_csv(TRAINING_HISTORY_CSV, index=False)
oof_true_df = pd.concat(oof_true_parts, ignore_index=True)
oof_predictions = pd.concat(oof_prediction_parts, ignore_index=True)
oof_predictions.to_csv(OOF_PREDICTIONS_CSV, index=False)
validation_metrics = evaluate_predictions(oof_true_df, oof_predictions)
print("Overall OOF metrics:")
print(json.dumps(validation_metrics, ensure_ascii=False, indent=2))
'''


TRAIN_ARTIFACT = r'''# ==========================================
# 5. Save config and optionally upload
# ==========================================

def save_inference_config():
    config = {
        "artifact_version": 8,
        "model_type": "ckip_bert_mlm_lora_shared_mlp",
        "model_name": MODEL_NAME,
        "backbone_path": MLM_BACKBONE_DIR.name,
        "ensemble_members": [f"fold_{fold}" for fold in FOLDS],
        "tokenizer_path": TOKENIZER_DIR.name,
        "max_len": MAX_LEN,
        "head_ratio": HEAD_RATIO,
        "task_classes": TASK_CLASSES,
        "thresholds": {
            "t1_yes": T1_THRESHOLD,
            "t3_yes": T3_THRESHOLD,
        },
        "multiclass_decision": "argmax",
        "ensemble": "mean_softmax_probability",
        "loss": {
            "type": "weighted_cross_entropy",
            "class_weight_formula": "(max_count / class_count) ** 0.5",
            "max_class_weight": MAX_CLASS_WEIGHT,
            "task_aggregation": "equal_mean",
        },
        "architecture": {
            "pooling": "cls",
            "shared_mlp": "hidden_size -> 256",
            "task_heads": "256 -> 128 -> output",
            "shared_dropout": 0.10,
            "head_dropout": 0.10,
        },
        "classification_training": {
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "min_f1_improvement": MIN_F1_IMPROVEMENT,
            "lora_learning_rate": LORA_LEARNING_RATE,
            "head_learning_rate": HEAD_LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "checkpoint_metric": "competition_macro_f1",
            "competition_score_weights": COMPETITION_SCORE_WEIGHTS,
        },
        "mlm": mlm_config,
        "fold_metrics": fold_metrics,
        "validation_metrics": validation_metrics,
    }
    with open(INFERENCE_CONFIG_JSON, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    return config


def validate_artifacts():
    required = [
        MLM_BACKBONE_DIR / "config.json",
        TOKENIZER_DIR / "tokenizer_config.json",
        MLM_CONFIG_JSON,
        MLM_HISTORY_CSV,
        TRAINING_HISTORY_CSV,
        OOF_PREDICTIONS_CSV,
        INFERENCE_CONFIG_JSON,
    ]
    missing = [str(path) for path in required if not path.exists()]
    backbone_weights = [
        MLM_BACKBONE_DIR / "model.safetensors",
        MLM_BACKBONE_DIR / "pytorch_model.bin",
    ]
    if not any(path.exists() for path in backbone_weights):
        missing.append("mlm_backbone model weights")
    for fold in FOLDS:
        fold_dir = OUTPUT_DIR / f"fold_{fold}"
        for path in [
            fold_dir / "adapter" / "adapter_config.json",
            fold_dir / "heads.pt",
            fold_dir / "metadata.json",
        ]:
            if not path.exists():
                missing.append(str(path))
        adapter_weights = [
            fold_dir / "adapter" / "adapter_model.safetensors",
            fold_dir / "adapter" / "adapter_model.bin",
        ]
        if not any(path.exists() for path in adapter_weights):
            missing.append(f"fold_{fold} adapter weights")
    if missing:
        raise FileNotFoundError("Missing artifacts: " + ", ".join(missing))


def push_artifacts_to_hf():
    validate_artifacts()
    api = HfApi()
    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type="model",
        private=HF_PRIVATE_REPO,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(OUTPUT_DIR),
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Upload simplified CKIP-BERT MLM + LoRA model",
    )
    print(f"Uploaded to https://huggingface.co/{HF_REPO_ID}")


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
# Simplified five-fold artifact v8 inference
# ==========================================

DEFAULT_REPO_ID = "maxbeettww/VeriPromise_ESG_2026_9906"
BATCH_SIZE = 16
ID_COLUMN = "id"
TEXT_COLUMN = "data"
TARGET_COLUMNS = [
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]
TASK_CLASSES = {
    "t1": ["No", "Yes"],
    "t2": [
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "more_than_5_years",
    ],
    "t3": ["No", "Yes"],
    "t4": ["Clear", "Not Clear", "Misleading"],
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


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
            hf_hub_download(repo_id=repo_id, filename=config_filename)
        except Exception:
            config_filename = "mtl_outputs/mtl_inference_config.json"
            hf_hub_download(repo_id=repo_id, filename=config_filename)
        downloaded = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[
                config_filename,
                "mlm_backbone/**",
                "tokenizer/**",
                "fold_*/adapter/**",
                "fold_*/heads.pt",
                "mtl_outputs/mlm_backbone/**",
                "mtl_outputs/tokenizer/**",
                "mtl_outputs/fold_*/adapter/**",
                "mtl_outputs/fold_*/heads.pt",
            ],
        )
        root = find_artifact_root(downloaded)
        if root is None:
            raise FileNotFoundError("Downloaded artifact has no config.")
        return root

    model_dir = Path("mtl_outputs") if model_dir is None else Path(model_dir)
    root = find_artifact_root(model_dir)
    if root is None:
        raise FileNotFoundError(f"No artifact found under {model_dir}.")
    return root


def load_config(root):
    with open(
        Path(root) / "mtl_inference_config.json",
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)
    if int(config.get("artifact_version", 0)) != 8:
        raise ValueError("This notebook only supports simplified artifact v8.")
    return config


def tokenize_head_tail(text, tokenizer, max_len, head_ratio):
    body_ids = tokenizer.encode(
        str(text),
        add_special_tokens=False,
        verbose=False,
    )
    max_body_len = max_len - 2
    if len(body_ids) > max_body_len:
        head_len = int(max_body_len * head_ratio)
        body_ids = (
            body_ids[:head_len]
            + body_ids[-(max_body_len - head_len):]
        )
    input_ids = [tokenizer.cls_token_id] + body_ids + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    pad_len = max_len - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len
    return input_ids, attention_mask


class InferenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, head_ratio):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_ratio = head_ratio

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        input_ids, attention_mask = tokenize_head_tail(
            self.dataframe.iloc[index][TEXT_COLUMN],
            self.tokenizer,
            self.max_len,
            self.head_ratio,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class SimpleESGModel(nn.Module):
    def __init__(self, backbone_path, adapter_dir):
        super().__init__()
        base_model = AutoModel.from_pretrained(backbone_path)
        self.backbone = PeftModel.from_pretrained(
            base_model,
            adapter_dir,
            is_trainable=False,
        )
        hidden_size = base_model.config.hidden_size
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Dropout(0.10),
                    nn.Linear(128, len(labels)),
                )
                for task, labels in TASK_CLASSES.items()
            }
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        features = self.shared_mlp(hidden[:, 0, :])
        return {
            task: head(features)
            for task, head in self.heads.items()
        }

    def load_heads(self, path):
        state = torch.load(path, map_location=DEVICE, weights_only=True)
        self.shared_mlp.load_state_dict(state["shared_mlp"])
        self.heads.load_state_dict(state["heads"])


def predict_probabilities(model, loader):
    output = {task: [] for task in TASK_CLASSES}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                logits = model(input_ids, attention_mask)
            for task in TASK_CLASSES:
                output[task].append(
                    torch.softmax(logits[task], dim=-1).cpu().numpy()
                )
    return {
        task: np.concatenate(parts, axis=0)
        for task, parts in output.items()
    }


def ensemble_probabilities(root, config, loader):
    members = config.get("ensemble_members", [])
    if len(members) != 5:
        raise ValueError("Artifact v8 must contain five ensemble members.")
    accumulated = {task: None for task in TASK_CLASSES}
    for member in members:
        member_dir = Path(root) / member
        print(f"Loading {member}")
        model = SimpleESGModel(
            Path(root) / config["backbone_path"],
            member_dir / "adapter",
        ).to(DEVICE)
        model.load_heads(member_dir / "heads.pt")
        probabilities = predict_probabilities(model, loader)
        for task, values in probabilities.items():
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


def route_predictions(dataframe, probabilities, t1_threshold, t3_threshold):
    rows = []
    for index, row in dataframe.reset_index(drop=True).iterrows():
        t1 = (
            "Yes"
            if probabilities["t1"][index, 1] >= t1_threshold
            else "No"
        )
        if t1 == "No":
            rows.append(
                {
                    ID_COLUMN: row[ID_COLUMN],
                    "promise_status": "No",
                    "verification_timeline": "N/A",
                    "evidence_status": "N/A",
                    "evidence_quality": "N/A",
                }
            )
            continue
        t2 = TASK_CLASSES["t2"][
            int(probabilities["t2"][index].argmax())
        ]
        t3 = (
            "Yes"
            if probabilities["t3"][index, 1] >= t3_threshold
            else "No"
        )
        if t3 == "No":
            rows.append(
                {
                    ID_COLUMN: row[ID_COLUMN],
                    "promise_status": "Yes",
                    "verification_timeline": t2,
                    "evidence_status": "No",
                    "evidence_quality": "N/A",
                }
            )
            continue
        rows.append(
            {
                ID_COLUMN: row[ID_COLUMN],
                "promise_status": "Yes",
                "verification_timeline": t2,
                "evidence_status": "Yes",
                "evidence_quality": TASK_CLASSES["t4"][
                    int(probabilities["t4"][index].argmax())
                ],
            }
        )
    return pd.DataFrame(rows)[[ID_COLUMN] + TARGET_COLUMNS]


def validate_input(dataframe):
    missing = [
        column
        for column in [ID_COLUMN, TEXT_COLUMN]
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"Test CSV missing columns: {missing}")
    if dataframe[ID_COLUMN].duplicated().any():
        raise ValueError("Test CSV contains duplicated ids.")


def inference_and_export(
    repo_id,
    test_csv_path,
    output_csv_path="final_submission.csv",
    model_dir=None,
):
    root = resolve_artifact_root(repo_id=repo_id, model_dir=model_dir)
    config = load_config(root)
    test_df = pd.read_csv(test_csv_path).reset_index(drop=True)
    validate_input(test_df)

    tokenizer = AutoTokenizer.from_pretrained(
        Path(root) / config["tokenizer_path"],
        use_fast=True,
    )
    loader = DataLoader(
        InferenceDataset(
            test_df,
            tokenizer,
            int(config["max_len"]),
            float(config["head_ratio"]),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=USE_AMP,
    )
    probabilities = ensemble_probabilities(root, config, loader)
    thresholds = config["thresholds"]
    output_df = route_predictions(
        test_df,
        probabilities,
        t1_threshold=float(thresholds["t1_yes"]),
        t3_threshold=float(thresholds["t3_yes"]),
    )
    output_df.to_csv(output_csv_path, index=False)
    print(f"Exported predictions to {output_csv_path}")
    print(output_df.head())
    return output_df
'''


INFERENCE_RUN = r'''# ==========================================
# Run inference
# ==========================================

TEST_CSV_PATH = "../data/ori_data/vpesg4k_test_2000.csv"

inference_and_export(
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
                "# CKIP-BERT MLM + LoRA Classification\n\n"
                "A minimal pipeline: ESG masked language modeling, five-fold "
                "LoRA classifiers, a shared MLP, four task MLP heads, "
                "class-weighted cross-entropy, probability averaging, and "
                "fixed inference decisions.\n"
            ),
            code(
                "# Colab dependencies. Restart the runtime if requested.\n"
                "# !pip install -q transformers peft accelerate torch pandas "
                "numpy scikit-learn tqdm huggingface_hub safetensors\n"
            ),
            code(TRAIN_CONFIG),
            code(TRAIN_DATA),
            code(TRAIN_MLM),
            code(TRAIN_CLASSIFIER),
            code(TRAIN_RUN),
            code(TRAIN_ARTIFACT),
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
                "# CKIP-BERT Simplified Inference\n\n"
                "Loads one ESG-MLM backbone and five LoRA classifiers, then "
                "averages their probabilities. T1/T3 use fixed 0.5 "
                "thresholds; T2/T4 use argmax.\n"
            ),
            code(
                "# Colab dependencies. Restart the runtime if requested.\n"
                "# !pip install -q transformers peft accelerate torch pandas "
                "numpy tqdm huggingface_hub safetensors\n"
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
