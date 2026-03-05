import pandas as pd
import json
import time
from openai import OpenAI
from tqdm import tqdm

# 1. 初始化設定
# 請填入你的 OpenAI API Key
client = OpenAI(api_key="sk-your-openai-api-key") 
INPUT_CSV = "train_data.csv"       # 原始 1000 筆資料的路徑
OUTPUT_CSV = "augmented_misleading_data.csv" # 輸出的擴充資料路徑

# 2. 定義 Prompt 模板
SYSTEM_PROMPT = """你現在是一位頂尖的 ESG 審計專家，深諳企業「漂綠（Greenwashing）」技巧。
任務：將提供的真實 ESG 樣本，改寫為具備「Misleading（誤導性）」證據的樣本。
要求：保持『承諾語句』不變，但將整段文本與『證據語句』竄改為誤導性（如指標劫持、偷換概念、因果斷裂）。
強制輸出格式：你必須輸出一個 JSON 物件，包含一個名為 "samples" 的陣列，陣列內包含 3 個不同誤導特徵的物件。
JSON 欄位必須完全對齊：["data", "promise_string", "evidence_string"]
"""

def build_user_prompt(row):
    return f"""
    請根據以下真實樣本，生成 3 組不同特徵的 Misleading 擴寫樣本：
    ESG 類型: {row['esg_type']}
    原始文本: {row['data']}
    承諾語句: {row['promise_string']}
    證據語句: {row['evidence_string']}
    """

# 3. 呼叫 LLM 的核心函數 (具備容錯機制)
def generate_misleading_samples(row, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(row)}
                ],
                response_format={ "type": "json_object" }, # 強制 JSON 輸出
                temperature=0.7
            )
            
            # 解析 JSON
            content = response.choices[0].message.content
            parsed_json = json.loads(content)
            return parsed_json.get("samples", [])
            
        except Exception as e:
            print(f"嘗試 {attempt+1} 失敗 (ID: {row['id']}): {e}")
            time.sleep(2) # 避免 Rate Limit
            
    return [] # 若重試皆失敗，回傳空陣列

# 4. 主執行管線
def main():
    print("讀取原始資料...")
    df = pd.read_csv(INPUT_CSV)
    
    # 篩選高品質的 Clear 樣本作為擴寫基底 (假設挑選 25 筆，即可生成 75 筆 Misleading)
    # 過濾條件：包含證據、且證據清晰、且文字長度大於一定字數避免太短的無意義句
    seed_samples = df[(df['evidence_quality'] == 'Clear') & 
                      (df['evidence_status'] == 'Yes')].sample(n=25, random_state=42)
    
    augmented_rows = []
    synthetic_id_counter = 90001 # 給予合成資料獨立的 ID 區段
    
    print("開始呼叫 LLM 進行擴寫...")
    for index, row in tqdm(seed_samples.iterrows(), total=len(seed_samples)):
        generated_variants = generate_misleading_samples(row)
        
        for variant in generated_variants:
            # 建立新的 DataFrame Row，繼承原有的 Metadata
            new_row = row.copy()
            new_row['id'] = synthetic_id_counter
            
            # 覆寫被 LLM 竄改的核心欄位
            new_row['data'] = variant.get('data', row['data'])
            new_row['promise_string'] = variant.get('promise_string', row['promise_string'])
            new_row['evidence_string'] = variant.get('evidence_string', row['evidence_string'])
            
            # 強制鎖定標籤為 Misleading
            new_row['evidence_quality'] = 'Misleading'
            new_row['promise_status'] = 'Yes'
            new_row['evidence_status'] = 'Yes'
            
            augmented_rows.append(new_row)
            synthetic_id_counter += 1
            
    # 5. 合併與儲存
    aug_df = pd.DataFrame(augmented_rows)
    print(f"\n成功生成 {len(aug_df)} 筆 Misleading 樣本。")
    
    # 儲存為獨立的 CSV，**不要**直接和原訓練集 concat！我們要在 K-Fold 階段動態注入。
    aug_df.to_csv(OUTPUT_CSV, index=False)
    print(f"擴寫資料已儲存至 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()