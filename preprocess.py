import pandas as pd
import re
import os
import csv

# Directory settings
base_dir = os.path.dirname(os.path.abspath(__file__))
output_700 = os.path.join(base_dir, "processed_700.csv")
output_9937 = os.path.join(base_dir, "processed_9937.csv")

# Regular expression to clean physical exam phrase "tim đều. phổi trong. bụng mềm"
clean_pattern = re.compile(r'tim\s+đều[\s.,-]*phổi\s+trong[\s.,-]*bụng\s+mềm[\s.,-]*', re.IGNORECASE)

# Regular expression to extract NIHSS score
nihss_pattern = re.compile(r'nihs{1,4}[:\s-]*(\d+)', re.IGNORECASE)

def preprocess_dataframe(df):
    processed_rows = []
    
    # 1. Fill NaNs with empty string
    text_cols = ['LYDO', 'HB_BENHLY', 'HB_BANTHAN', 'KB_TOANTHAN', 'KB_BOPHAN', 'MAICD']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
        else:
            df[col] = ""

    for idx, row in df.iterrows():
        lydo = row['LYDO']
        hb_benhly = row['HB_BENHLY']
        hb_banthan = row['HB_BANTHAN']
        kb_toanthan = row['KB_TOANTHAN']
        kb_bophan = row['KB_BOPHAN']
        maicd = row['MAICD']

        # 2. Create Label based on MAICD containing 'I63'
        label = 1 if 'i63' in maicd.lower() else 0

        # 3. Text cleaning: remove "tim đều. phổi trong. bụng mềm"
        kb_toanthan_clean = clean_pattern.sub("", kb_toanthan).strip()
        kb_bophan_clean = clean_pattern.sub("", kb_bophan).strip()

        # 4. Extract NIHSS score from all 5 original text columns combined
        combined_text_for_nihss = f"{lydo} {hb_benhly} {hb_banthan} {kb_toanthan} {kb_bophan}"
        nihss_match = nihss_pattern.search(combined_text_for_nihss)
        if nihss_match:
            score = nihss_match.group(1)
            nihss_str = f"Thang điểm đột quỵ NIHSS là {score}."
        else:
            nihss_str = "Không ghi nhận đánh giá NIHSS."

        # 5. Concat Input_Text
        input_text = (
            f"Lý do: {lydo} | "
            f"Bệnh sử: {hb_benhly} | "
            f"Tiền sử: {hb_banthan} | "
            f"Toàn thân: {kb_toanthan_clean} | "
            f"Bộ phận: {kb_bophan_clean} | "
            f"{nihss_str}"
        )

        processed_rows.append({
            'Input_Text': input_text,
            'Label': label
        })
        
    return pd.DataFrame(processed_rows)

def process_pipeline():
    try:
        # Load Group 1: 700_co.xlsx and 700_khong.xlsx
        print("Reading Group 1 (700_co.xlsx, 700_khong.xlsx)...")
        df_700_co = pd.read_excel(os.path.join(base_dir, "700_co.xlsx"))
        df_700_khong = pd.read_excel(os.path.join(base_dir, "700_khong.xlsx"))
        
        print("Preprocessing Group 1...")
        proc_700_co = preprocess_dataframe(df_700_co)
        proc_700_khong = preprocess_dataframe(df_700_khong)
        
        # Combine Group 1
        combined_700 = pd.concat([proc_700_co, proc_700_khong], ignore_index=True)
        
        # Export processed_700.csv
        combined_700.to_csv(output_700, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print(f"Exported: {output_700}")
        print(f"Total rows in processed_700.csv: {len(combined_700)}")
        
    except Exception as e:
        print(f"Error processing Group 1: {e}")

    try:
        # Load Group 2: 9937_co.xlsx and 9937_khong.xlsx
        print("\nReading Group 2 (9937_co.xlsx, 9937_khong.xlsx)...")
        df_9937_co = pd.read_excel(os.path.join(base_dir, "9937_co.xlsx"))
        df_9937_khong = pd.read_excel(os.path.join(base_dir, "9937_khong.xlsx"))
        
        print("Preprocessing Group 2...")
        proc_9937_co = preprocess_dataframe(df_9937_co)
        proc_9937_khong = preprocess_dataframe(df_9937_khong)
        
        # Combine Group 2
        combined_9937 = pd.concat([proc_9937_co, proc_9937_khong], ignore_index=True)
        
        # Export processed_9937.csv
        combined_9937.to_csv(output_9937, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print(f"Exported: {output_9937}")
        print(f"Total rows in processed_9937.csv: {len(combined_9937)}")
        
    except Exception as e:
        print(f"Error processing Group 2: {e}")

if __name__ == "__main__":
    process_pipeline()
