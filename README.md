# VeriPromiseESG 2026 TEAM_9906

VeriPromiseESG 是一個 ESG 承諾驗證分類任務。本專案針對企業永續報告中的純文字片段，預測四個下游分類欄位，並輸出符合提交格式的 CSV。

四個分類任務：

1. **promise_status**：是否存在可驗證的 ESG 承諾，標籤為 `Yes` / `No`。
2. **verification_timeline**：承諾驗證時間，標籤為 `already`、`within_2_years`、`between_2_and_5_years`、`longer_than_5_years`。
3. **evidence_status**：是否提供證據，標籤為 `Yes` / `No`。
4. **evidence_quality**：證據品質，標籤為 `Clear`、`Not Clear`、`Misleading`。

提交 CSV 欄位固定為：

```text
id,promise_status,verification_timeline,evidence_status,evidence_quality
```

Routing 規則固定與原始架構一致：

- `promise_status=No` 時，`verification_timeline`、`evidence_status`、`evidence_quality` 輸出 `N/A`。
- `promise_status=Yes` 且 `evidence_status=No` 時，`evidence_quality` 輸出 `N/A`。

## Project Structure

```text
VeriPromiseESG_2026_TEAM_9906/
|-- app/
|   |-- data/
|   |   |-- ori_data/
|   |   |   |-- vpesg4k_train_1000 V1.csv
|   |   |   `-- augmented_misleading_data.csv
|   |   |-- clean_data/
|   |   |   |-- train_fold_1.csv
|   |   |   |-- ...
|   |   |   `-- val_fold_5.csv
|   |   |-- analysis_results/
|   |   |-- ori_dataset_analysis.py
|   |   |-- data_argument.py
|   |   `-- argument_combine.py
|   `-- model/
|       |-- model_train.ipynb
|       |-- model_inference.ipynb
|       |-- model_setfit.ipynb
|       |-- model_setfit_inference.ipynb
|       `-- compare_result.py
|-- final_submission.csv
|-- README.md
`-- LICENSE
```

## Modeling Methods

### Original Transformer MTL

原始流程位於 `app/model/model_train.ipynb` 與 `app/model/model_inference.ipynb`。

核心設計：

- 使用 `ckiplab/bert-base-chinese` 作為 Transformer backbone。
- 使用共享 backbone 與四個任務 head。
- 透過 masked multi-task loss 保留 T1 到 T4 的階層關係。
- 使用 5-fold 訓練與 fold soft-voting 推論。
- 使用固定 routing 產生 `final_submission.csv`。

`model_inference.ipynb` 只支援原始 Transformer MTL artifact：

```text
best_mtl_model_fold_1.pth
...
best_mtl_model_fold_5.pth
tokenizer/
```

它不會讀取 SetFit 的 `encoder/` 或 `task_heads.joblib`，因此 SetFit 模型必須使用 `model_setfit_inference.ipynb`。

### Multi-Task SetFit

SetFit 訓練流程位於 `app/model/model_setfit.ipynb`。SetFit 推論流程位於 `app/model/model_setfit_inference.ipynb`。

核心設計：

- 預設 pre-trained model 使用 `BAAI/bge-base-zh-v1.5`。
- 使用 SetFit-style contrastive learning fine-tune 共享 SentenceTransformer encoder。
- T1-T4 共用同一個 embedding space，但保留四個 task-specific heads。
- 訓練時加入 task prefix，避免不同任務的 `Yes` / `No` 標籤互相混淆。
- synthetic `Misleading` 只用於 T4 head 與 T4 contrastive view，不用來強化 T1-T3。
- encoder 最多訓練 10 epochs，每個 epoch 使用固定 validation pairs 計算 cosine average precision。
- validation pairs 不含 synthetic data，也不使用 model-dependent hard-negative mining。
- validation cosine AP 連續 2 次沒有至少 `0.0001` 的改善時提前停止。
- 訓練完成後自動載入並保存 validation cosine AP 最佳的 encoder，而不是最後一個 epoch。
- 在 OOF validation 上搜尋 T1/T3 threshold，以最大化四個任務的平均 macro F1。

`model_setfit.ipynb` 會輸出：

```text
setfit_outputs/
|-- README.md
|-- setfit_thresholds.json
|-- setfit_inference_config.json
|-- setfit_oof_predictions.csv
|-- setfit_oof_probabilities.csv
|-- fold_1/
|   |-- encoder/
|   `-- task_heads.joblib
|-- ...
`-- fold_5/
    |-- encoder/
    `-- task_heads.joblib
```

Early-stopping checkpoint 只會暫存在訓練期間，完成後自動刪除。正式 artifact 路徑與既有 inference 契約不變：

- 最佳 encoder：`setfit_outputs/fold_{fold}/encoder/`
- task heads：`setfit_outputs/fold_{fold}/task_heads.joblib`
- thresholds：`setfit_outputs/setfit_thresholds.json`
- inference config：`setfit_outputs/setfit_inference_config.json`

### Choose The Correct Inference Notebook

| 訓練流程 | 推論 notebook | 模型 artifact |
|---|---|---|
| Original Transformer MTL | `app/model/model_inference.ipynb` | `best_mtl_model_fold_{fold}.pth`、`tokenizer/` |
| Multi-Task SetFit | `app/model/model_setfit_inference.ipynb` | `fold_{fold}/encoder/`、`fold_{fold}/task_heads.joblib` |

兩條流程都接受至少包含 `id,data,esg_type` 的 CSV，並輸出相同欄位順序：

```text
id,promise_status,verification_timeline,evidence_status,evidence_quality
```

## End-To-End Workflow

以下流程假設資料前處理與 fold 拆分在本機完成，訓練 notebook 手動上傳到 Google Colab 執行。

### 1. Prepare Data Locally

在本機安裝資料處理依賴：

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openai python-dotenv tqdm
```

若要重新產生 LLM augmentation，請在 `app/data/.env` 設定：

```text
API_KEY=your_openai_api_key
```

進入資料處理資料夾：

```bash
cd app/data
```

可選：執行 EDA。

```bash
python ori_dataset_analysis.py
```

可選：重新產生 synthetic `Misleading` 資料。

```bash
python data_argument.py
```

建立 5-fold 訓練與驗證 CSV：

```bash
python argument_combine.py
```

輸出位置：

```text
app/data/clean_data/train_fold_1.csv
app/data/clean_data/train_fold_2.csv
app/data/clean_data/train_fold_3.csv
app/data/clean_data/train_fold_4.csv
app/data/clean_data/train_fold_5.csv
app/data/clean_data/val_fold_1.csv
app/data/clean_data/val_fold_2.csv
app/data/clean_data/val_fold_3.csv
app/data/clean_data/val_fold_4.csv
app/data/clean_data/val_fold_5.csv
```

`argument_combine.py` 只把 synthetic data 放進 train folds，不放進 validation folds，以避免 validation leakage。

### 2. Upload Fold CSVs To Colab

在 Colab 建立 runtime 後，上傳：

- `app/model/model_setfit.ipynb`
- `app/data/clean_data/train_fold_1.csv` 到 `train_fold_5.csv`
- `app/data/clean_data/val_fold_1.csv` 到 `val_fold_5.csv`

建議在 Colab 建立同樣的資料夾：

```python
import os
os.makedirs("/content/app/data/clean_data", exist_ok=True)
```

再把十個 fold CSV 上傳到：

```text
/content/app/data/clean_data/
```

### 3. Train SetFit On Colab

在 Colab 開啟 `model_setfit.ipynb`，若使用手動上傳的 CSV，設定：

```python
USE_GITHUB_RAW = False
```

若直接讀取 GitHub raw fold CSV，維持：

```python
USE_GITHUB_RAW = True
```

目前 primary 與 fallback model 都設定為 `BAAI/bge-base-zh-v1.5`。如需更換 encoder，請同時確認 Colab GPU 記憶體與 SentenceTransformer 相容性。

encoder 訓練的防 overfitting 設定如下：

```python
CONTRASTIVE_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.0001
```

每個 fold 都會顯示 training loss、validation loss、validation cosine AP、最佳 epoch 與實際停止 epoch。只有最佳 encoder 會寫入正式 artifact 目錄。

依序執行 notebook。訓練完成後會產生：

- `setfit_outputs/`
- `setfit_final_submission.csv`

### 4. Upload SetFit Artifacts To Hugging Face

`model_setfit.ipynb` 末段提供 Hugging Face Hub 上傳流程，參考原本 `model_train.ipynb` 的做法。

在設定 cell 中填寫或保留：

```python
RUN_HF_UPLOAD = True
HF_SETFIT_REPO_ID = None
HF_PRIVATE_REPO = False
```

若 `HF_SETFIT_REPO_ID = None`，notebook 會使用登入帳號建立：

```text
{your_hf_username}/VeriPromise_ESG_2026_9906_SetFit
```

若要指定 repo，例如：

```python
HF_SETFIT_REPO_ID = "maxbeettww/VeriPromise_ESG_2026_9906_SetFit"
```

執行上傳 cell 後，notebook 會：

1. 執行 `notebook_login()`。
2. 建立或重用 Hugging Face model repo。
3. 上傳整個 `setfit_outputs/` 內容到 repo 根目錄。

上傳後 Hugging Face repo 應包含：

```text
fold_1/encoder/
fold_1/task_heads.joblib
...
fold_5/encoder/
fold_5/task_heads.joblib
setfit_thresholds.json
setfit_inference_config.json
README.md
```

### 5. Run SetFit Inference From Hugging Face

在 Colab 開啟 `app/model/model_setfit_inference.ipynb`。

預設 repo id：

```python
DEFAULT_REPO_ID = "maxbeettww/VeriPromise_ESG_2026_9906_SetFit"
```

如果你的上傳 repo 不同，請改成實際 repo id。

推論 notebook 會：

1. 使用 `snapshot_download()` 從 Hugging Face Hub 下載 SetFit artifact。
2. 讀取 `setfit_inference_config.json`。
3. 載入 5-fold shared encoders 與 task heads。
4. 做 5-fold probability averaging。
5. 套用 `setfit_thresholds.json` 的 T1/T3 threshold。
6. 使用原本 routing 輸出 `final_submission.csv`。

Early stopping 不會改變任何 inference 輸入或 artifact schema，因此不需要修改 `model_setfit_inference.ipynb`，重新訓練後可直接載入新的最佳 encoder。

測試 CSV 至少需要包含：

```text
id,data,esg_type
```

預設測試輸入為 `val_fold_1.csv`，方便和原本流程比較。若要對真正測試集推論，請把測試 CSV 上傳到 Colab，並改成：

```python
test_url = "/content/test.csv"
ensemble_inference_and_export(DEFAULT_REPO_ID, test_url, "final_submission.csv")
```

輸出檔：

```text
final_submission.csv
```

## Evaluation

原始比較腳本位於：

```text
app/model/compare_result.py
```

原腳本主要計算 weighted F1。SetFit notebook 內另外提供 macro F1、weighted F1、per-class F1 與 confusion matrix，因為目前優化目標是四個下游任務的平均 macro F1。

若要在本機比較原始模型與 SetFit 模型，可分別保留：

```text
final_submission.csv
setfit_final_submission.csv
```

並依需要調整 `compare_result.py` 的 `file2` 路徑。

## Notes

- 不建議把 T1-T4 完全分開訓練，因為四個任務具有明確前後依賴。
- 不建議把 T1-T4 合成單一 joint label classifier，因為正式輸出與評估是四個獨立欄位。
- 目前採用折衷設計：共享 SetFit encoder，加上四個 task-specific heads。
- SetFit inference notebook 的輸入/輸出格式與原本 `model_inference.ipynb` 一致。
- `model_inference.ipynb` 與 `model_setfit_inference.ipynb` 的模型 artifact 不相容，請依訓練流程選擇對應 notebook。

---

Developed by **TEAM_9906** for the VeriPromiseESG 2026 Challenge.
