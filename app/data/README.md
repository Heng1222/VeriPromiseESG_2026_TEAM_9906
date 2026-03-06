# VeriPromiseESG 資料處理與增強模組

本目錄包含了 VeriPromiseESG 專案的資料分析、合成資料增強以及 K-Fold 交叉驗證切分的完整工作流程。主要目標是針對原始 1000 筆 ESG 樣本進行深入分析，並利用大型語言模型（LLM）生成誤導性（Misleading）樣本以解決類別不平衡問題，最後進行嚴格的資料隔離切分。

## 📁 目錄結構

*   `ori_data/`: 存放原始資料集。
    *   `vpesg4k_train_1000 V1.csv`: 官方提供的 1000 筆真實訓練資料。
    *   `augmented_misleading_data.csv`: 由 LLM 生成的合成誤導性樣本。
*   `clean_data/`: 存放經過處理與切分後的資料。
    *   `train_fold_[1-5].csv`: 包含真實資料與合成資料的訓練集。
    *   `val_fold_[1-5].csv`: 僅包含真實資料的純淨驗證集，確保評估無偏誤。
*   `analysis_results/`: 存放由 `ori_dataset_analysis.py` 產生的各類 EDA 統計圖表（如分佈圖、缺失值熱圖等）。
*   `.env`: 存放 API 金鑰等敏感環境變數（請勿上傳至版本控制系統）。

## ⚙️ 工作流程與腳本說明

請依照以下順序執行腳本以完成資料準備工作：

### 1. 探索性資料分析 (EDA)
**執行腳本：** `python ori_dataset_analysis.py`
*   **功能：** 自動分析原始資料集的欄位型態、缺失值狀況，並針對數值型、類別型與文字型欄位進行視覺化。
*   **輸出：** 統計圖表將儲存在 `analysis_results/` 資料夾中。

### 2. 資料增強 (Data Augmentation)
**執行腳本：** `python data_argument.py`
*   **功能：** 
    *   讀取原始資料中高品質的「Clear」樣本作為基底。
    *   利用 OpenAI LLM 將真實樣本竄改為具備「誤導性 (Misleading)」特徵的合成資料。
    *   保持承諾語句不變，但修改證據語句以呈現指標劫持或因果斷裂等漂綠特徵。
*   **輸出：** `ori_data/augmented_misleading_data.csv`。

### 3. K-Fold 切分與動態注入
**執行腳本：** `python argument_combine.py`
*   **功能：**
    *   採用 **Proxy Stratification (降維分層)** 技術，解決樣本極端不平衡導致的切分困難。
    *   執行 **5-Fold 交叉驗證切分**。
    *   **動態注入機制**：僅將合成資料注入「訓練集 (Train Fold)」，而「驗證集 (Val Fold)」保持 100% 真實資料，絕對防止資料洩漏 (Data Leakage)。
*   **輸出：** `clean_data/` 下的 10 個 CSV 檔案。

## 🛠️ 環境需求

*   Python 3.8+
*   必要套件：`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `openai`, `python-dotenv`, `tqdm`
*   需在 `.env` 檔案中設定 `API_KEY` 以供 LLM 呼叫。

## ⚠️ 注意事項

*   **資料純淨性**：在進行模型評估時，務必確保驗證集中不含任何合成資料（ID > 90000），本工作流已透過 `argument_combine.py` 自動確保此點。
*   **API 成本**：執行 `data_argument.py` 會產生 LLM 呼叫費用，請根據需求調整 seed samples 的數量。
