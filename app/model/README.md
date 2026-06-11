# Model

此目錄包含 VeriPromiseESG 的模型訓練、推論與評估程式。

## Notebooks

| 檔案 | 用途 | 模型 artifact |
|---|---|---|
| `model_train.ipynb` | CKIP-BERT multi-task 5-fold fine-tuning 與 OOF 評估 | `mtl_outputs/` |
| `model_inference.ipynb` | CKIP-BERT 5-fold probability ensemble 推論 | `mtl_outputs/` 或 Hugging Face repo |
| `model_setfit.ipynb` | Multi-Task SetFit 5-fold 訓練與 OOF 評估 | encoder、task heads、thresholds |
| `model_setfit_inference.ipynb` | SetFit 5-fold soft-voting 推論 | `setfit_outputs/` 或 Hugging Face repo |

CKIP-BERT 與 SetFit 的外部 CSV 介面相同，但 artifact 不相容，必須使用各自對應的 inference notebook。

## Input And Output Compatibility

CKIP-BERT 與 SetFit 都不會改變外部輸入或提交格式。

### Training Input

`model_train.ipynb` 與 `model_setfit.ipynb` 都讀取：

```text
app/data/clean_data/train_fold_1.csv
...
app/data/clean_data/train_fold_5.csv
app/data/clean_data/val_fold_1.csv
...
app/data/clean_data/val_fold_5.csv
```

訓練資料必須包含：

```text
id,data,promise_status,verification_timeline,evidence_status,evidence_quality
```

其他欄位如 `esg_type`、`company`、`page_number` 可以保留，但兩套模型都只要求 `data` 作為文字輸入，不依賴測試資料中的額外欄位。

### Inference Input

`model_inference.ipynb` 與 `model_setfit_inference.ipynb` 的測試 CSV 最少只要求：

```text
id,data
```

- `id` 不可重複。
- `esg_type` 可以存在，但目前不會加入模型輸入。
- 輸入列數與 `id` 順序會原樣保留。

### Prediction Output

輸出欄位與順序維持不變：

```text
id,promise_status,verification_timeline,evidence_status,evidence_quality
```

Routing 規則也維持不變：

- `promise_status=No` 時，T2、T3、T4 輸出 `N/A`。
- `promise_status=Yes` 且 `evidence_status=No` 時，T4 輸出 `N/A`。
- T2 的正式標籤使用 `longer_than_5_years`；舊資料中的 `more_than_5_years` 會在訓練時正規化。

獨立 inference notebook 預設輸出 `final_submission.csv`，兩套模型的 CSV schema 相同。

## CKIP-BERT Fine-Tuning

`model_train.ipynb` 保留共享 CKIP-BERT backbone、四個既有 MLP heads，以及 masked BCE/CE loss。主要改善集中在訓練與模型選擇，不建立 SetFit pairs，也不增加新文字：

- Backbone 與 tokenizer 統一使用 `ckiplab/bert-base-chinese`。
- 只使用 `data`，訓練與推論共用相同的 head-tail token truncation。
- 五個 folds 各訓練一個 encoder；每個 fold 只保留 validation competition macro-F1 最高的 epoch。
- Competition 權重固定為 T1 `0.20`、T2 `0.15`、T3 `0.30`、T4 `0.35`。
- Backbone 與 heads 使用不同 learning rate，並加入 warmup、cosine decay、gradient clipping、AMP、gradient accumulation 與 early stopping。
- T1-T4 使用平方根反頻率 class weights，降低 T2 `within_2_years` 與 T4 `Misleading` 的類別不平衡影響。
- 既有 synthetic `Misleading` rows 只計入 T4，sample weight 固定為 `0.35`，不再污染 T1-T3。
- 完成五 folds 後，以 2,000 筆 OOF probabilities 搜尋 T1/T3 routing thresholds。
- 推論時先對五個 folds 的 probabilities 做平均，再套用 thresholds 與階層 routing。

這個設計仍保留 validation。直接用全部 2,000 筆訓練而完全不驗證，無法選 epoch、threshold 或偵測過擬合；五個 fold 的 ensemble 則能讓每筆訓練資料參與四個模型，同時保留完整 OOF 評估。

### CKIP-BERT Artifacts

```text
mtl_outputs/
|-- fold_1/
|   |-- best_model.pth
|   `-- training_history.csv
|-- ...
|-- fold_5/
|   |-- best_model.pth
|   `-- training_history.csv
|-- tokenizer/
|-- mtl_inference_config.json
|-- mtl_thresholds.json
|-- mtl_oof_probabilities.csv
|-- mtl_oof_predictions.csv
`-- training_history.csv
```

本機推論：

```python
ensemble_inference_and_export(
    repo_id=None,
    test_csv_path="/content/test.csv",
    output_csv_path="final_submission.csv",
    model_dir="/content/mtl_outputs",
)
```

Hugging Face artifact：

```python
ensemble_inference_and_export(
    repo_id="maxbeettww/VeriPromise_ESG_2026_9906_CKIP_MTL",
    test_csv_path="/content/test.csv",
    output_csv_path="final_submission.csv",
)
```

## Updated SetFit Training

`model_setfit.ipynb` 只使用一組固定設定，最多訓練五個 shared encoders，每個 fold 一次，不執行多組 encoder 實驗。

### Shared Encoder

- Backbone：`BAAI/bge-base-zh-v1.5`
- T1-T4 共用 SentenceTransformer encoder。
- 最多 8 epochs，batch size 16，learning rate `2e-5`。
- Checkpoint 明確依官方競賽權重計算 validation pair AP：
  - T1：`0.20`
  - T2：`0.15`
  - T3：`0.30`
  - T4：`0.35`
- Notebook 會驗證儲存的 best checkpoint 確實對應最高 `weighted_ap`，避免誤用 validation loss。

### Pair Sampling

- 所有 contrastive pairs 禁止 self-pair 與重複 pair。
- T1、T3 保留平衡正負 pairs，並各加入 240 個 hard negatives。
- T2 使用不平衡感知採樣：
  - `within_2_years` 使用最多 378 個唯一 positive pairs。
  - 每類 500 個 negative pairs。
  - 相鄰時間區間獲得較高負樣本抽樣權重。
  - 額外加入 600 個相鄰區間 hard negatives。
- T4 positive pairs：
  - `Clear=600`
  - `Not Clear=600`
  - `Misleading=300`
- T4 額外加入 800 個指定類別邊界 hard negatives。
- 現有 synthetic `Misleading` rows 仍保留；沒有新增資料列或新文本。

### Task Heads

四個任務都依該 fold validation macro F1 選擇分類頭，不只最佳化 T2/T4。

- Logistic regression：比較 balanced 與平方根反頻率 class weight。
- Calibrated LinearSVC：比較相同兩種 class weight。
- T2 額外比較 ordinal cumulative logistic head。
- T4 額外比較 MLP head。
- T4 synthetic rows 的 head sample weight 固定為 `0.35`。

完成五 folds 後，只搜尋 T1/T3 routing thresholds，目標為官方 competition weighted score。

## Validation And Quality Gate

訓練完成後會輸出：

- 每個 fold 的 T1-T4 macro F1 與 weighted F1。
- 每個 fold 的 per-class report 與 confusion matrix。
- Pooled OOF metrics、predictions 與 probabilities。
- 官方權重 competition score。

T1/T3 設有回歸檢查：

- 參考 T1 OOF macro F1：`0.766654`
- 參考 T3 OOF macro F1：`0.709650`
- 任一任務下降超過 `0.01` 時，quality gate 失敗並禁止 Hugging Face upload。

Notebook 程式已通過 pair、ordinal probability、JSON 與語法測試，但更新後的實際 5-fold GPU 分數必須重新執行訓練後才能取得。

## SetFit Artifacts

訓練輸出：

```text
setfit_outputs/
|-- fold_1/
|   |-- encoder/
|   `-- task_heads.joblib
|-- ...
|-- fold_5/
|   |-- encoder/
|   `-- task_heads.joblib
|-- setfit_thresholds.json
|-- setfit_inference_config.json
|-- setfit_oof_predictions.csv
|-- setfit_oof_probabilities.csv
`-- README.md
```

外部 CSV 介面沒有改變，但 artifact 內部版本已更新為 `artifact_version=2`：

- `task_heads.joblib` 可能包含 T2 ordinal head。
- `setfit_inference_config.json` 記錄 weighted AP、pair profile、head candidates 與 quality gate。
- 更新後的 `model_setfit_inference.ipynb` 可讀取舊版 v1 與新版 v2 config。
- 新版 ordinal artifact 必須使用更新後的 inference notebook。

## Inference

本機 artifact：

```python
ensemble_inference_and_export(
    repo_id=None,
    test_csv_path="/content/test.csv",
    output_csv_path="final_submission.csv",
    model_dir="/content/setfit_outputs",
)
```

Hugging Face artifact：

```python
ensemble_inference_and_export(
    repo_id="maxbeettww/VeriPromise_ESG_2026_9906_SetFit",
    test_csv_path="/content/test.csv",
    output_csv_path="final_submission.csv",
)
```

Inference 會執行五個 fold 的 probability soft voting，再套用 T1/T3 thresholds 與原有 routing 規則。
