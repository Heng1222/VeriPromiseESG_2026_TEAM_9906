# Model

## Simplified CKIP-BERT Pipeline

`model_train.ipynb` now contains only:

1. ESG masked language modeling.
2. Five CKIP-BERT LoRA classifiers.
3. One shared MLP.
4. Four task-specific MLP heads.
5. Class-weighted cross-entropy loss.
6. Fixed inference decisions.

Five-fold training and probability ensembling are retained. Temperature
calibration, class bias tuning, threshold search, focal loss, ordinal loss,
pair loss, synthetic pairing, and synthetic holdout logic are removed.

## Data

### MLM

MLM uses only the `data` column from:

```text
app/data/ori_data/vpesg4k_train_1000 V1.csv
app/data/ori_data/vpesg4k_val_1000.csv
app/data/ori_data/vpesg4k_test_2000.csv
```

This gives 4,000 documents. Argument/synthetic data is excluded.

MLM settings:

- Backbone: `ckiplab/bert-base-chinese`
- Maximum length: 512
- Overflow overlap: 64 tokens
- Dynamic masking: 15%
- Epochs: 6
- Batch size: 4
- Gradient accumulation: 4
- Learning rate: `3e-5`
- Scheduler: 8% warmup followed by cosine decay

### Classification

Classification uses the existing prepared folds:

```text
app/data/clean_data/train_fold_1.csv
...
app/data/clean_data/train_fold_5.csv
app/data/clean_data/val_fold_1.csv
...
app/data/clean_data/val_fold_5.csv
```

Each train fold contains 1,600 official rows and 111 prepared synthetic
`Misleading` rows. Each validation fold contains 400 official rows. Synthetic
rows are treated as ordinary classification samples; there is no special
pairing, holdout, loss, or routing rule.

## Model

```text
ESG MLM backbone
  -> LoRA
  -> CLS representation
  -> shared MLP: hidden_size -> 256
  -> four task heads: 256 -> 128 -> output
```

Each task uses weighted cross-entropy. Every fold derives its weights from its
own training split:

```text
weight = min((largest_class_count / class_count) ** 0.5, 5.0)
```

The square-root weighting raises rare-class importance without the instability
of full inverse-frequency weighting. Losses from tasks that have valid labels
in the current batch are averaged equally.

One model is trained for each fold for at most 30 epochs and stops after 5
non-improving epochs. The checkpoint score uses the official competition
weights: T1 20%, T2 15%, T3 30%, and T4 35%.

LoRA parameters use a `5e-5` learning rate while the newly initialized shared
MLP and task heads use `1e-4`. Both groups use 8% warmup and cosine decay.
This keeps the longer run conservative for the pretrained backbone while
allowing the classification layers to learn faster. Inference averages the
five models' softmax probabilities before applying the fixed prediction
rules.

## Prediction Rules

- T1 `promise_status`: predict `Yes` when `P(Yes) >= 0.5`.
- T2 `verification_timeline`: use softmax argmax.
- T3 `evidence_status`: predict `Yes` when `P(Yes) >= 0.5`.
- T4 `evidence_quality`: use softmax argmax.
- If T1 is `No`, T2-T4 are `N/A`.
- If T1 is `Yes` and T3 is `No`, T4 is `N/A`.

A threshold of 0.5 is appropriate for the binary T1 and T3 tasks. T2 and T4
are multiclass tasks, so argmax is the normal fixed decision rule.

## Artifact

```text
mtl_outputs/
|-- mlm_backbone/
|-- tokenizer/
|-- fold_1/
|   |-- adapter/
|   |-- heads.pt
|   `-- metadata.json
|-- ...
|-- fold_5/
|-- mlm_config.json
|-- mlm_training_history.csv
|-- training_history.csv
|-- oof_predictions.csv
`-- mtl_inference_config.json
```

Artifact v8 restores the smaller CLS-only head after the dual-pooling v7
experiment regressed. It is intentionally incompatible with v7 head weights.

## Inference

Local artifact:

```python
inference_and_export(
    repo_id=None,
    test_csv_path="../data/ori_data/vpesg4k_test_2000.csv",
    output_csv_path="final_submission.csv",
    model_dir="mtl_outputs",
)
```

Hugging Face artifact:

```python
inference_and_export(
    repo_id="maxbeettww/VeriPromise_ESG_2026_9906",
    test_csv_path="../data/ori_data/vpesg4k_test_2000.csv",
    output_csv_path="final_submission.csv",
)
```

Input:

```text
id,data
```

Output:

```text
id,promise_status,verification_timeline,evidence_status,evidence_quality
```

## Maintenance

Regenerate notebooks:

```powershell
uv run python app/model/build_ckip_lora_notebooks.py
```

Run contract tests:

```powershell
uv run python -m unittest app.model.test_ckip_lora_notebooks -v
```

The SetFit notebooks remain separate and use their own artifact format.
