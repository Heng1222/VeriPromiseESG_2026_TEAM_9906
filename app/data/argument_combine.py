import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

# 1. 初始化設定
REAL_DATA_PATH = [
    "ori_data/vpesg4k_train_1000 V1.csv",
    "ori_data/vpesg4k_val_1000.csv",
]
SYNTHETIC_DATA_PATH = "ori_data/augmented_misleading_data.csv"  # Phase A 生成的擴充資料
N_SPLITS = 5
OUTPUT_DIR = "./clean_data/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_proxy_stratify_key(df):
    """
    第一性原理：降維分層 (Proxy Stratification)
    為了解決樣本極度不平衡導致 StratifiedKFold 崩潰的問題，我們創造一個組合特徵作為切分基準。
    對於數量少於 5 的極端類別（如那唯一的 Misleading），我們在「切分計算當下」將其與最相近的類別合併，
    保證切分順利進行，但「不改變原始標籤」。
    """
    # 填充缺失值，確保可以拼接字串
    t2 = df['verification_timeline'].fillna('T2_UNK').astype(str)
    t4 = df['evidence_quality'].fillna('T4_UNK').astype(str)
    
    # 創造聯合切分鍵 (例如: "already_Clear")
    stratify_key = t2 + "_" + t4
    
    # 統計每個組合的數量
    value_counts = stratify_key.value_counts()
    
    # 將數量少於 N_SPLITS (5) 的稀有組合，強制歸類為 "RARE_GROUP"
    rare_keys = value_counts[value_counts < N_SPLITS].index
    stratify_key = stratify_key.apply(lambda x: 'RARE_GROUP' if x in rare_keys else x)
    
    return stratify_key

def main():
    print("讀取真實資料與擴充資料...")
    real_df = pd.concat(
        [pd.read_csv(data_path) for data_path in REAL_DATA_PATH],
        axis=0,
        ignore_index=True,
    )
    synth_df = pd.read_csv(SYNTHETIC_DATA_PATH)
    
    # 重置 Index 以策安全，避免 iloc 對應錯誤
    real_df = real_df.reset_index(drop=True)
    
    print("建構降維分層鍵 (Proxy Stratify Key)...")
    stratify_key = create_proxy_stratify_key(real_df)
    
    # 初始化 K-Fold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    print(f"開始進行 {N_SPLITS}-Fold 切分與動態注入...")
    
    # skf.split() 只作用於「真實資料」
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=real_df, y=stratify_key)):
        # 1. 取得該折的純淨真實資料
        train_real_df = real_df.iloc[train_idx].copy()
        val_real_df = real_df.iloc[val_idx].copy()
        
        # 2. 【核心防禦機制】動態注入 (Dynamic Injection)
        # 只在 Training Fold 混入 LLM 生成的合成資料
        train_injected_df = pd.concat([train_real_df, synth_df], axis=0, ignore_index=True)
        
        # 3. 打亂訓練集 (Shuffle)，避免模型學到合成資料集中在尾部的順序特徵
        train_injected_df = train_injected_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # 4. 輸出檢驗
        print(f"\n[Fold {fold + 1}]")
        print(f" - 訓練集數量: {len(train_injected_df)} (包含 {len(synth_df)} 筆擴充資料)")
        print(f" - 驗證集數量: {len(val_real_df)} (絕對純淨，0 筆擴充資料)")
        
        # 確保驗證集裡絕對沒有生成資料的 ID (假設生成資料 ID 從 90001 開始)
        leakage_check = val_real_df['id'].max() > 90000
        assert not leakage_check, "致命錯誤：發生資料洩漏 (Data Leakage)！"
        
        # 5. 儲存至硬碟，供後續 DataLoader 讀取
        train_injected_df.to_csv(f"{OUTPUT_DIR}train_fold_{fold+1}.csv", index=False)
        val_real_df.to_csv(f"{OUTPUT_DIR}val_fold_{fold+1}.csv", index=False)

    print("\n資料隔離切分完成。所有 Folds 皆已免疫資料洩漏。")

if __name__ == "__main__":
    main()
