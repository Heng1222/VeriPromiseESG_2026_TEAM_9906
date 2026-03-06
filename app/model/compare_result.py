import pandas as pd
from sklearn.metrics import f1_score


ID_COLUMN = "id"

TARGET_COLUMNS = [
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]

MISSING_ALLOWED_COLUMNS = [
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]

ALL_COLUMNS = [ID_COLUMN] + TARGET_COLUMNS


def load_and_validate_csv(file_path):
    """讀取 CSV，檢查必要欄位與重複 id。"""
    df = pd.read_csv(file_path)

    missing_cols = [col for col in ALL_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{file_path} 缺少必要欄位: {missing_cols}")

    df = df[ALL_COLUMNS].copy()

    if df[ID_COLUMN].duplicated().any():
        duplicated_ids = df.loc[df[ID_COLUMN].duplicated(), ID_COLUMN].tolist()
        raise ValueError(f"{file_path} 存在重複 id，例如: {duplicated_ids[:10]}")

    df[ID_COLUMN] = df[ID_COLUMN].astype(int)
    return df


def normalize_value(value, allow_missing=False):
    """
    將欄位值標準化。
    - allow_missing=True 時，'N/A'、空字串、NaN 都視為缺值
    - allow_missing=False 時，保留一般字串類別
    """
    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if allow_missing:
        if value == "" or value.upper() == "N/A":
            return pd.NA

    return value


def normalize_column(series, allow_missing=False):
    """把整個欄位做標準化。"""
    return series.apply(lambda x: normalize_value(x, allow_missing=allow_missing))


def compare_results(file1, file2):
    print(f"正在讀取檔案1: {file1}")
    df_true = load_and_validate_csv(file1)

    print(f"正在讀取檔案2: {file2}")
    df_pred = load_and_validate_csv(file2)

    # 先標準化欄位
    for col in TARGET_COLUMNS:
        allow_missing = col in MISSING_ALLOWED_COLUMNS
        df_true[col] = normalize_column(df_true[col], allow_missing=allow_missing)
        df_pred[col] = normalize_column(df_pred[col], allow_missing=allow_missing)

    # 用 id 對齊
    df_merged = pd.merge(
        df_true,
        df_pred,
        on=ID_COLUMN,
        suffixes=("_true", "_pred")
    )

    if len(df_merged) != len(df_true) or len(df_merged) != len(df_pred):
        print("警告：兩個檔案的 id 沒有完全對齊")
        print(f"file1 筆數: {len(df_true)}")
        print(f"file2 筆數: {len(df_pred)}")
        print(f"merge 後筆數: {len(df_merged)}")

        ids_true = set(df_true[ID_COLUMN])
        ids_pred = set(df_pred[ID_COLUMN])

        only_in_file1 = sorted(list(ids_true - ids_pred))
        only_in_file2 = sorted(list(ids_pred - ids_true))

        if only_in_file1:
            print(f"只存在於 file1 的 id（前 10 筆）: {only_in_file1[:10]}")
        if only_in_file2:
            print(f"只存在於 file2 的 id（前 10 筆）: {only_in_file2[:10]}")

    print("\n--- 各欄位 Weighted F1 Score ---")

    results = {}

    for col in TARGET_COLUMNS:
        true_col = f"{col}_true"
        pred_col = f"{col}_pred"

        # 這三欄允許缺值，所以只評 true / pred 都非缺值的列
        if col in MISSING_ALLOWED_COLUMNS:
            valid_mask = df_merged[true_col].notna() & df_merged[pred_col].notna()
        else:
            # promise_status 不應缺值；若有缺值，也同樣略過避免報錯
            valid_mask = df_merged[true_col].notna() & df_merged[pred_col].notna()

        df_eval = df_merged.loc[valid_mask, [true_col, pred_col]].copy()

        total_count = len(df_merged)
        valid_count = len(df_eval)
        skipped_count = total_count - valid_count

        print(f"\n欄位: {col}")
        print(f"可評分筆數: {valid_count} / {total_count}")
        if skipped_count > 0:
            print(f"跳過缺值筆數: {skipped_count}")

        if valid_count == 0:
            print("無可用資料，無法計算 F1 Score")
            results[col] = None
            continue

        y_true = df_merged[true_col].fillna("N/A")
        y_pred = df_merged[pred_col].fillna("N/A")

        score = f1_score(y_true, y_pred, average="weighted")
        results[col] = score

        print(f"Weighted F1 Score: {score:.4f}")

    valid_scores = [score for score in results.values() if score is not None]

    print("\n--- Summary ---")
    for col, score in results.items():
        if score is None:
            print(f"{col}: N/A")
        else:
            print(f"{col}: {score:.4f}")

    if valid_scores:
        avg_f1 = sum(valid_scores) / len(valid_scores)
        print(f"Average Weighted F1 Score: {avg_f1:.4f}")
    else:
        print("Average Weighted F1 Score: N/A")

def main():
    # 手動指定檔案
    file1 = "../data/clean_data/val_fold_1.csv"  # 真實標籤
    file2 = "../../final_submission.csv"
    
    compare_results(file1, file2)


if __name__ == "__main__":
    main()