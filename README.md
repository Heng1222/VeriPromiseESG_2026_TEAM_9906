# VeriPromiseESG: 漂綠行為識別與 ESG 承諾驗證系統

Developed by **TEAM_9906** for the VeriPromiseESG 2026 Challenge.

## 🌟 專案簡介

VeriPromiseESG 致力於利用人工智慧技術自動化審查企業的 ESG（環境、社會、公司治理）承諾。本專案開發了一套完整的流程，從原始資料分析、合成資料增強，到深度學習模型訓練與評估，旨在精確識別企業揭露中的「漂綠 (Greenwashing)」行為。

主要目標是針對四個關鍵指標進行分類預測：
1.  **Promise Status**: 承諾狀態識別。
2.  **Verification Timeline**: 驗證時間線分析。
3.  **Evidence Status**: 證據充足性評估。
4.  **Evidence Quality**: 證據品質鑑定。

## 🚀 核心技術亮點

*   **LLM 驅動的資料增強**: 利用 OpenAI GPT 模型針對原始樣本進行「對抗式竄改」，生成具備指標劫持、因果斷裂等特徵的誤導性樣本，解決類別不平衡問題。
*   **動態注入 K-Fold 策略**: 採用 5-Fold 交叉驗證，並確保合成資料僅進入訓練集，驗證集保持 100% 真實資料，嚴格防止資料洩漏。
*   **多標題分類架構**: 基於 Transformers 框架建構深度學習模型，針對 ESG 文字特徵進行微調，優化加權 F1 分數。

## 📂 目錄架構

```text
VeriPromiseESG_2026_TEAM_9906/
├── app/
│   ├── data/                 # 資料處理與增強模組
│   │   ├── clean_data/       # 經過切分與增強後的 Fold 資料
│   │   ├── ori_data/         # 原始資料與生成的合成資料
│   │   ├── analysis_results/ # EDA 統計圖表輸出
│   │   ├── data_argument.py  # LLM 資料增強腳本
│   │   ├── argument_combine.py # K-Fold 切分與資料合成腳本
│   │   └── ori_dataset_analysis.py # 探索性資料分析
│   └── model/                # 模型訓練與推論模組
│       ├── model_train.ipynb # 模型訓練實驗室
│       ├── model_inference.ipynb # 推論與生成提交檔
│       └── compare_result.py # 結果比對與效能評估工具
├── img/                      # 說明文件用圖檔
├── final_submission.csv      # 最終預測結果
├── README.md                 # 專案總覽
└── LICENSE                   # 授權條款
```

## 🛠️ 執行流程說明

### 步驟 1: 資料準備與分析
1.  **安裝依賴**: `pip install pandas numpy scikit-learn matplotlib seaborn openai python-dotenv tqdm transformers torch`
2.  **環境設定**: 在 `app/data/` 下建立 `.env` 並填入 `API_KEY`。
3.  **執行 EDA**: 執行 `python app/data/ori_dataset_analysis.py` 生成資料統計圖表。
4.  **資料增強**: 執行 `python app/data/data_argument.py` 產生合成誤導性樣本。
5.  **K-Fold 切分**: 執行 `python app/data/argument_combine.py` 產生五折訓練/驗證集。

### 步驟 2: 模型開發
1.  **模型訓練**: 開啟 `app/model/model_train.ipynb`，載入 `clean_data` 中的 Fold 進行微調。
2.  **模型推論**: 使用 `app/model/model_inference.ipynb` 對測試集進行預測並匯出 `final_submission.csv`。

### 步驟 3: 效能驗證
*   使用 `app/model/compare_result.py` 評估預測結果與驗證集標籤的一致性，計算 Weighted F1 Score。

## 📊 評估指標

本專案主要優化目標為 **Average Weighted F1 Score**，綜合評估上述四個目標欄位的預測準確度。

---
© 2026 TEAM_9906. Developed for VeriPromiseESG Challenge.
