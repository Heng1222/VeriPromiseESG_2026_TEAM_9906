# Model

此目錄包含 VeriPromiseESG 的模型訓練、推論、評估與 notebook 產生工具。

## Notebooks

| 檔案 | 用途 | 模型 artifact |
|---|---|---|
| `model_train.ipynb` | CKIP-BERT RSLoRA multi-task 訓練、5-fold OOF 校準與 full-data ensemble | `mtl_outputs/`，artifact v3 |
| `model_inference.ipynb` | CKIP-BERT v3 ensemble 推論，並相容舊版 v1/v2 checkpoint | `mtl_outputs/` 或 Hugging Face repo |
| `model_setfit.ipynb` | Multi-Task SetFit 5-fold 訓練與 OOF 評估 | `setfit_outputs/` |
| `model_setfit_inference.ipynb` | SetFit 5-fold soft-voting 推論 | `setfit_outputs/` 或 Hugging Face repo |

CKIP-BERT 與 SetFit 的外部 CSV 介面相同，但 artifact 不相容，必須使用各自對應的 inference notebook。

## Input And Output

### Training Data

新版 `model_train.ipynb` 讀取：

```text
app/data/ori_data/vpesg4k_train_1000 V1.csv
app/data/ori_data/vpesg4k_val_1000.csv
app/data/ori_data/augmented_misleading_data.csv
app/data/clean_data/val_fold_1.csv
...
app/data/clean_data/val_fold_5.csv
```

兩份官方 labeled data 合併為 2,000 筆真實資料。`val_fold_*.csv` 只用來取得既有 fold ID，不直接當成另一份資料加入。111 筆 synthetic `Misleading` 資料只參與 T4 與 paired contrastive loss。

必要欄位：

```text
id,data,promise_status,verification_timeline,evidence_status,evidence_quality
```

Synthetic 配對使用 `pdf_url` 與 `promise_string`；source group 則使用 `pdf_url` 與 `page_number`。目前 111 筆中有 105 筆可配回原始 Clear 樣本；37 個 synthetic source groups 會完整分配到單一 fold，避免同來源變體同時出現在訓練與 auxiliary validation。

### Inference Input

`model_inference.ipynb` 最少只要求：

```text
id,data
```

- `id` 不可重複。
- 其他欄位可以存在，但不會加入模型輸入。
- 輸出列數與 `id` 順序和輸入完全一致。

### Prediction Output

```text
id,promise_status,verification_timeline,evidence_status,evidence_quality
```

Routing 規則：

- `promise_status=No` 時，T2、T3、T4 輸出 `N/A`。
- `promise_status=Yes` 且 `evidence_status=No` 時，T4 輸出 `N/A`。
- 官方 T2 長期標籤是 `more_than_5_years`。
- 新版 CKIP-BERT inference 會把舊資料或舊 artifact 中的 `longer_than_5_years` 轉成官方值後匯出。

輸出前會檢查欄位順序、合法標籤、routing、列數與 ID 順序。預設輸出檔名為 `final_submission.csv`。

## CKIP-BERT LoRA Training

### Model

- Backbone：`ckiplab/bert-base-chinese`。
- 使用 PEFT RSLoRA，套用所有 Transformer linear layers。
- LoRA 設定：`r=8`、`alpha=16`、dropout `0.1`、`use_rslora=True`。
- 文字只使用 `data`，訓練與推論共用 head-tail truncation，最大長度 512。
- Pooling：`CLS + masked mean + masked max`，合併後維度為 2304。
- Shared MLP：`2304 -> 384 -> 256`。
- 四個 task heads：`256 -> 128 -> output`，全部使用 multi-sample dropout。

### Objectives

- T1-T4 使用 effective-number class-balanced focal loss，`gamma=1.5`，class weight 上限為 6。
- Competition task weights：
  - T1 `0.20`
  - T2 `0.15`
  - T3 `0.30`
  - T4 `0.35`
- T2 額外加入權重 `0.15` 的 ordinal expected-distance loss。
- Synthetic T4 sample weight 為 `0.25`。
- 可配對的 synthetic/原始 Clear 樣本加入權重 `0.05` 的 cosine-margin loss。
- Synthetic rows 不參與 T1-T3 loss。

### Optimization

- LoRA learning rate：`1e-4`。
- Shared MLP 與 task heads learning rate：`3e-4`。
- Batch size 8，gradient accumulation 2，effective batch size 16。
- 最多 12 epochs，early stopping patience 3。
- Warmup 10%、cosine decay、AMP 與 gradient clipping 1.0。

## Training Flow

### 1. Five-Fold OOF

每個 fold：

1. 使用約 1,600 筆真實資料與四份 synthetic source groups 訓練。
2. 使用 400 筆真實 validation 選擇 competition macro-F1 最佳 epoch。
3. 保留一份 synthetic source groups 作 auxiliary holdout。
4. 儲存 LoRA adapter、MLP heads、training history 與 OOF logits。

完成五 folds 後：

- 使用 OOF logits 為四個任務分別估計 scalar temperature。
- T1/T3 threshold 以 `mean fold score - 0.25 * fold std` 為搜尋目標。
- `Misleading` 同時使用 probability threshold 與相對於 Clear/Not Clear 的 margin。
- `Misleading` threshold 必須令真實非 Misleading OOF false-positive rate 不超過 0.5%，再最大化 synthetic holdout recall。

### 2. Full-Data Ensemble

- 取五個 fold 最佳 epoch 的中位數作 full-data 訓練 epoch。
- 使用全部 2,000 筆真實資料與 111 筆 synthetic 資料。
- 分別使用 seeds `42`、`123`、`2026` 訓練三個模型。
- 最終 v3 inference 只平均這三個 full-data members 的 calibrated probabilities。
- 每個 member 儲存後會重新載入，執行 logits save/load parity 檢查。

## Validation And Quality Gate

訓練 notebook 會輸出：

- Overall OOF competition score。
- 官方 train 1,000、官方 val 1,000 與每個 fold 的 task macro F1。
- Per-class classification report、confusion matrix 與 prediction distribution。
- T4 direct/routed macro F1。
- 兩筆真實 `Misleading` 的 probabilities 與 prediction。
- Synthetic holdout recall 與真實非 Misleading false-positive rate。

Quality gate 基準來自舊版 OOF：

| 指標 | 基準 |
|---|---:|
| Competition | `0.601140` |
| T1 | `0.754363` |
| T2 | `0.565558` |
| T3 | `0.724977` |
| T4 | `0.422687` |

要求：

- Competition 不得低於 `0.601140`。
- T1、T2、T3 不得比各自基準下降超過 `0.02`。
- T4 必須改善，或維持基準且降低 `Misleading` false-positive rate。
- Quality gate 失敗時 notebook 會停止，不會進入 full-data ensemble 與 artifact upload。

目前 repository 只完成程式、靜態測試與資料契約驗證；新版 GPU 訓練尚未執行，因此沒有新版實際 F1。

## CKIP-BERT Artifact V3

```text
mtl_outputs/
|-- fold_1/
|   |-- adapter/
|   |-- heads.pt
|   |-- metadata.json
|   `-- training_history.csv
|-- ...
|-- fold_5/
|-- full_seed_42/
|   |-- adapter/
|   |-- heads.pt
|   |-- metadata.json
|   `-- training_history.csv
|-- full_seed_123/
|-- full_seed_2026/
|-- tokenizer/
|-- mtl_inference_config.json
|-- mtl_calibration.json
|-- mtl_thresholds.json
|-- mtl_oof_logits.csv
|-- mtl_oof_probabilities.csv
|-- mtl_oof_predictions.csv
|-- mtl_synthetic_holdout_logits.csv
`-- training_history.csv
```

`mtl_inference_config.json` 會記錄：

- artifact version 與 base model。
- 三個 ensemble members。
- LoRA、pooling 與 task class 設定。
- temperatures、routing thresholds 與 `Misleading` thresholds。
- quality metrics 與 synthetic policy。

## CKIP-BERT Inference

本機 artifact：

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
    repo_id="maxbeettww/VeriPromise_ESG_2026_9906",
    test_csv_path="/content/test.csv",
    output_csv_path="final_submission.csv",
)
```

Inference notebook 支援：

- v3：下載並 ensemble 三個 LoRA full-data members。
- v1/v2：沿用舊版五個 full checkpoint probability ensemble。
- 遠端 artifact 會先讀取 config 判斷版本；v3 不會下載同 repo 中的大型 legacy checkpoints。

上傳前在 training notebook 將：

```python
RUN_HF_UPLOAD = True
```

預設為 `False`，避免訓練完成前覆寫既有 Hugging Face artifact。

## Notebook Maintenance And Tests

兩個 CKIP-BERT notebook 由可 review 的 Python 來源產生：

```powershell
uv run python app/model/build_ckip_lora_notebooks.py
```

修改產生器後必須重新執行，將內容同步到 `.ipynb`。

核心 contract 測試：

```powershell
uv run python -m unittest app.model.test_ckip_lora_notebooks -v
```

測試涵蓋：

- Notebook code cell 語法。
- Artifact v3 關鍵設定。
- 官方 timeline alias。
- T1/T3/T4 routing。
- `Misleading` probability + margin 決策。
- Submission schema 與合法值。
- 37 個 synthetic source groups 的 fold 隔離。
- 105 筆 synthetic/source 配對。

## SetFit

`model_setfit.ipynb` 與 `model_setfit_inference.ipynb` 維持既有 artifact v2 流程：

- Backbone：`BAAI/bge-base-zh-v1.5`。
- T1-T4 共用 SentenceTransformer encoder。
- 五個 fold probability soft voting。
- 每個 task head 依 validation macro F1 選擇。
- T2 支援 ordinal cumulative logistic head。
- T4 支援 MLP head 與 synthetic sample weighting。
- 推論套用 T1/T3 thresholds 與相同階層 routing。

已知相容性限制：目前 SetFit 內部及匯出仍使用 legacy T2 標籤 `longer_than_5_years`，尚未套用 CKIP-BERT v3 的官方 `more_than_5_years` 匯出正規化。提交 SetFit 結果前必須先修正或後處理該欄位。

SetFit artifact：

```text
setfit_outputs/
|-- fold_1/
|   |-- encoder/
|   `-- task_heads.joblib
|-- ...
|-- fold_5/
|-- setfit_thresholds.json
|-- setfit_inference_config.json
|-- setfit_oof_predictions.csv
|-- setfit_oof_probabilities.csv
`-- README.md
```

本機推論：

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
