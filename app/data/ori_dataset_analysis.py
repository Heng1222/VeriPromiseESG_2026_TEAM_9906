import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_dataset(file_path):
    """
    Performs a detailed Exploratory Data Analysis (EDA) on the given CSV dataset.

    Args:
        file_path (str): The path to the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'。請檢查檔案路徑是否正確。")
        return
    except Exception as e:
        print(f"讀取檔案時發生錯誤：{e}")
        return

    # Create a directory for saving plots
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("==================================================")
    print("                 數據集 EDA 分析報告                ")
    print("==================================================")
    print(f"檔案: {file_path}\n")

    print("\n--- 1. 資料集概覽 ---\n")
    print(f"資料筆數: {len(df)}")
    print(f"欄位數量: {len(df.columns)}")
    print("\n欄位列表:")
    print(df.columns.tolist())

    print("\n\n--- 2. 資料集前 5 筆預覽 ---\n")
    print(df.head())

    print("\n\n--- 3. 資料型態與缺失值 ---\n")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    info_df = pd.DataFrame({
        '資料型態': df.dtypes,
        '非空值數量': df.notnull().sum(),
        '缺失值數量': missing_values,
        '缺失值比例 (%)': missing_percentage.round(2)
    })
    print(info_df)
    
    # Visualize missing values
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Value Heatmap')
    plt.savefig(os.path.join(output_dir, 'missing_value_heatmap.png'))
    plt.close()

    print("\n\n--- 4. 各欄位詳細分析 ---\n")
    for col in df.columns:
        print(f"\n----- 分析欄位: '{col}' -----\\n")
        
        # --- 數值型欄位 ---
        if pd.api.types.is_numeric_dtype(df[col]):
            print("類型: 數值型")
            print("\n描述性統計:")
            print(df[col].describe())
            
            # Plot distribution
            plt.figure(figsize=(10, 5))
            sns.histplot(df[col].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, f'numeric_{col}_distribution.png'))
            plt.close()

        # --- 類別型欄位 (包含看起來像數值的類別, 如 ticker) ---
        elif pd.api.types.is_object_dtype(df[col]) and df[col].nunique() < 50 or col == 'ticker':
            print("類型: 類別型")
            value_counts = df[col].value_counts(dropna=False)
            value_percentages = df[col].value_counts(dropna=False, normalize=True).mul(100).round(2)
            
            category_df = pd.DataFrame({
                '數量': value_counts,
                '比例 (%)': value_percentages
            })
            
            print(f"\n類別數量: {df[col].nunique()}")
            print("各類別分布:")
            print(category_df)
            
            # Plot value counts
            plt.figure(figsize=(12, 7))
            sns.countplot(y=df[col], order=value_counts.index, palette='viridis')
            plt.title(f'Category Counts for {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'categorical_{col}_counts.png'))
            plt.close()


        # --- 文字型欄位 ---
        else:
            print("類型: 文字型")
            # Fill NA to handle calculations on missing text
            df_text = df[col].fillna('')
            text_lengths = df_text.str.len()
            
            print("\n文字長度分析:")
            print(f"  最短長度: {text_lengths.min()}")
            print(f"  最長長度: {text_lengths.max()}")
            print(f"  平均長度: {text_lengths.mean():.2f}")
            print(f"  中位數長度: {text_lengths.median()}")

            # Plot text length distribution
            plt.figure(figsize=(10, 5))
            sns.histplot(text_lengths, kde=True, bins=30)
            plt.title(f'Text Length Distribution for {col}')
            plt.xlabel('Text Length')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, f'text_{col}_length_distribution.png'))
            plt.close()

    print("\n==================================================")
    print("                   分析結束                   ")
    print(f"圖表已儲存至 '{output_dir}' 資料夾中。")
    print("==================================================")


if __name__ == "__main__":
    # 設定中文顯示
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei'] # 優先使用微軟正黑體, 備用簡體黑體
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
    file_name = "ori_data/vpesg4k_train_1000 V1.csv"
    analyze_dataset(file_name)
