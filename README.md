# VeriPromiseESG 2026 TEAM_9906

VeriPromiseESG 是一個 ESG 承諾驗證分類任務。本專案針對企業永續報告中的純文字片段，預測四個下游分類欄位，並輸出符合提交格式的 CSV。

四個分類任務如下：

1. **promise_status**：是否存在可驗證的 ESG 承諾，標籤為 `Yes` / `No`。
2. **verification_timeline**：承諾驗證時間，標籤為 `already`、`within_2_years`、`between_2_and_5_years`、`longer_than_5_years`。
3. **evidence_status**：是否提供證據，標籤為 `Yes` / `No`。
4. **evidence_quality**：證據品質，標籤為 `Clear`、`Not Clear`、`Misleading`。

模型輸出欄位固定為：

```text
id,promise_status,verification_timeline,evidence_status,evidence_quality
```

當 `promise_status=No` 時，後三欄輸出 `N/A`。當 `evidence_status=No` 時，`evidence_quality` 輸出 `N/A`。

## Project Structure

```text
VeriPromiseESG_2026_TEAM_9906/
├── app/
│   ├── data/
│   │   ├── ori_data/
│   │   │   ├── vpesg4k_train_1000 V1.csv
│   │   │   └── augmented_misleading_data.csv
│   │   ├── clean_data/
│   │   │   ├── train_fold_1.csv
│   │   │   ├── ...
│   │   │   └── val_fold_5.csv
│   │   ├── analysis_results/
│   │   ├── ori_dataset_analysis.py
│   │   ├── data_argument.py
│   │   └── argument_combine.py
│   └── model/
│       ├── model_train.ipynb
│       ├── model_inference.ipynb
│       ├── model_setfit.ipynb
│       └── compare_result.py
├── final_submission.csv
├── README.md
└── LICENSE
```

## Modeling Methods

### Original Transformer MTL

原始流程位於 `app/model/model_train.ipynb` 與 `app/model/model_inference.ipynb`。

核心設計：

- 使用 `ckiplab/bert-base-chinese` 作為 Transformer backbone。
- 使用一個共享 backbone 與四個任務 head。
- 透過自訂 masked multi-task loss 保留 T1 到 T4 的階層關係。
- 使用 5-fold 訓練與 fold soft-voting 推論。
- 使用固定 routing 產生最終 CSV。

### Multi-Task SetFit

新增流程位於 `app/model/model_setfit.ipynb`，不會修改原本的訓練與推論 notebook。

SetFit 版本的目標是改善小資料分類場景下的 macro F1，尤其是少數類別如 `within_2_years` 與 `Misleading`。

核心設計：

- 預設 pre-trained model 使用 `BAAI/bge-large-zh-v1.5`。
- 若 GPU 記憶體不足，可在 notebook 設定 `FORCE_FALLBACK_MODEL = True`，改用 `BAAI/bge-base-zh-v1.5`。
- 使用 SetFit-style contrastive learning fine-tune 一個共享 SentenceTransformer encoder。
- T1-T4 不完全分開訓練，而是共用同一個 embedding space。
- 為避免不同任務的 `Yes` / `No` 混淆，訓練時加入 task prefix，例如：
  - `任務：判斷是否有 ESG 承諾。`
  - `任務：判斷承諾驗證時間。`
  - `任務：判斷是否提供證據。`
  - `任務：判斷證據品質。`
- encoder fine-tune 完成後，固定共享 encoder，分別訓練四個 lightweight task heads。
- synthetic `Misleading` 資料只用於 T4 head 與 T4 contrastive view，不用來強化 T1-T3，避免污染上游任務。
- 在 OOF validation 上搜尋 T1/T3 threshold，以最大化四個任務的平均 macro F1。
- 輸出檔名為 `setfit_final_submission.csv`，避免覆蓋原本的 `final_submission.csv`。

## End-to-End Workflow

以下流程假設資料前處理與 fold 拆分在本機完成，訓練 notebook 手動上傳到 Google Colab 執行。

### 1. Prepare Data Locally

先在本機建立 Python 環境並安裝資料處理依賴：

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openai python-dotenv tqdm
```

若要重新產生 LLM augmentation，請在 `app/data/.env` 設定：

```text
API_KEY=your_openai_api_key
```

接著進入資料處理資料夾：

```bash
cd app/data
```

可選步驟：執行 EDA。

```bash
python ori_dataset_analysis.py
```

可選步驟：重新產生 synthetic `Misleading` 資料。

```bash
python data_argument.py
```

這會輸出：

```text
app/data/ori_data/augmented_misleading_data.csv
```

建立 5-fold 訓練與驗證 CSV：

```bash
python argument_combine.py
```

這會輸出：

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

注意：`argument_combine.py` 只把 synthetic data 放進 train folds，不放進 validation folds，以避免 validation leakage。

### 2. Upload Files To Colab

在 Colab 建立一個新的 runtime，然後手動上傳：

- `app/model/model_setfit.ipynb`
- `app/data/clean_data/train_fold_1.csv`
- `app/data/clean_data/train_fold_2.csv`
- `app/data/clean_data/train_fold_3.csv`
- `app/data/clean_data/train_fold_4.csv`
- `app/data/clean_data/train_fold_5.csv`
- `app/data/clean_data/val_fold_1.csv`
- `app/data/clean_data/val_fold_2.csv`
- `app/data/clean_data/val_fold_3.csv`
- `app/data/clean_data/val_fold_4.csv`
- `app/data/clean_data/val_fold_5.csv`

建議在 Colab 檔案區建立與本機一致的資料夾結構：

```text
/content/app/data/clean_data/
```

把十個 fold CSV 放到上述資料夾。若使用 Colab 左側 Files 面板手動上傳，也可以先在 notebook cell 執行：

```python
import os
os.makedirs("/content/app/data/clean_data", exist_ok=True)
```

再把 CSV 上傳到該資料夾。

### 3. Configure The SetFit Notebook

在 Colab 開啟 `model_setfit.ipynb` 後，先確認設定 cell。

若要使用手動上傳到 Colab 的 fold CSV，請設定：

```python
USE_GITHUB_RAW = False
```

若要直接讀取 GitHub raw fold CSV，則維持：

```python
USE_GITHUB_RAW = True
```

若 GPU 記憶體不足，請設定：

```python
FORCE_FALLBACK_MODEL = True
```

預設測試輸入為 `val_fold_1.csv`，方便和原本 `final_submission.csv` 比較：

```python
TEST_CSV_PATH = f"{RAW_BASE_URL}val_fold_1.csv"
```

若要對真正測試集推論，請把測試 CSV 上傳到 Colab，並改成該路徑，例如：

```python
TEST_CSV_PATH = "/content/test.csv"
```

測試 CSV 至少需要包含：

```text
id,data,esg_type
```

### 4. Run Training On Colab

在 Colab 依序執行 `model_setfit.ipynb` 全部 cells。

notebook 會執行：

1. 安裝 SetFit 與 sentence-transformers 依賴。
2. 讀取 5-fold CSV。
3. 建立 T1-T4 task-prefixed training views。
4. 使用 contrastive learning fine-tune shared encoder。
5. 為 T1-T4 訓練四個 task heads。
6. 儲存每個 fold 的模型到：

```text
setfit_outputs/fold_1/
setfit_outputs/fold_2/
setfit_outputs/fold_3/
setfit_outputs/fold_4/
setfit_outputs/fold_5/
```

7. 產生 OOF prediction 並計算：

- 每個任務 macro F1
- 每個任務 weighted F1
- average macro F1
- per-class F1
- confusion matrix

8. 在 OOF 上搜尋最佳 T1/T3 threshold。

### 5. Export Final Prediction CSV

notebook 最後會輸出：

```text
setfit_final_submission.csv
```

欄位順序固定為：

```text
id,promise_status,verification_timeline,evidence_status,evidence_quality
```

如果是在 Colab 執行，完成後可從左側 Files 面板下載 `setfit_final_submission.csv`。

## Evaluation

原始比較腳本位於：

```text
app/model/compare_result.py
```

原腳本主要計算 weighted F1。SetFit notebook 內另外提供 macro F1 評估，因為目前優化目標是四個下游分類任務的平均 macro F1。

若要在本機比較原本輸出與 SetFit 輸出，可把 `setfit_final_submission.csv` 放回專案根目錄，並依需要調整 `compare_result.py` 的 `file2` 路徑。

## Notes

- 不建議把 T1-T4 完全分開訓練，因為四個任務具有明確前後依賴。
- 不建議把 T1-T4 合成單一 joint label classifier，因為正式輸出與評估是四個獨立欄位。
- 目前採用折衷設計：共享 SetFit encoder，加上四個 task-specific heads。
- `setfit_final_submission.csv` 不會覆蓋原本的 `final_submission.csv`，方便比較兩種方法差異。

---

Developed by **TEAM_9906** for the VeriPromiseESG 2026 Challenge.
