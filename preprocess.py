import pandas as pd
import re
import os
import csv
import random

# Directory settings
base_dir = os.path.dirname(os.path.abspath(__file__))
output_700 = os.path.join(base_dir, "processed_700.csv")
output_9937 = os.path.join(base_dir, "processed_9937.csv")

# Regular expression to extract NIHSS score
nihss_pattern = re.compile(r'nihs{1,4}[:\s-]*(\d+)', re.IGNORECASE)

def advanced_clinical_text_cleaner(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Chuyển về chữ thường để dễ xử lý viết tắt
    text = text.lower()
    
    # 2. Xóa mẫu câu hành chính và câu khám nhiễu (đã chốt từ trước)
    text = re.sub(r"tim\s*đều[\s.,]*phổi\s*trong[\s.,]*bụng\s*mềm[\s.,]*", "", text)
    
    # Gộp mẫu của SIS và mẫu nv/nhập viện cuối câu
    boilerplate_pattern = r"(=*>|-*>|->)?\s*(?:(?:(nhập|chuyển)\s*(viện\s*)?(bệnh viện\s*|bv\s*)?(s\.i\.s|sis)(\s*cần thơ)?)|nv|nhập\s*viện)[\s.,]*$"
    text = re.sub(boilerplate_pattern, "", text)
    
    # 3. MỞ RỘNG VIẾT TẮT Y KHOA (Sử dụng \b để giới hạn từ độc lập)
    abbreviations = {
        r"\bbn\b": "bệnh nhân",
        r"\bbv\b": "bệnh viện",
        r"\bha\b": "huyết áp",
        r"\btha\b": "tăng huyết áp",
        r"\bđtđ\b": "đái tháo đường",
        r"\bclvt\b": "cắt lớp vi tính",
        r"\btbmmnt\b": "tai biến mạch máu não",
        r"\btb\b": "tai biến",
        r"\bnv\b": "nhập viện",
        r"\bđm\b": "động mạch",
        r"\bvltl\b": "vật lý trị liệu",
        r"\bphcn\b": "phục hồi chức năng",
        # Lưu ý với P và T (thường đứng sau bộ phận cơ thể hoặc dấu phẩy)
        r"\b(tay|chân|mắt|bên|người|chi|nửa|nữa)\s*[\s,.-]*\s*\(?p\)?(?!\w)": r"\1 phải",
        r"\b(tay|chân|mắt|bên|người|chi|nửa|nữa)\s*[\s,.-]*\s*\(?t\)?(?!\w)": r"\1 trái"
    }
    
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)
        
    # 3.1 SỬA LỖI CHÍNH TẢ "nữa" -> "nửa" (cho bộ phận / phương hướng)
    text = re.sub(r"\bnữa\s+(người|bên|đầu|thân|phải|trái)\b", r"nửa \1", text)
        
    # 4. CHUẨN HÓA KHOẢNG TRẮNG VÀ DẤU CÂU (Noise Cleaning)
    # Xóa dấu câu thừa trước gạch đứng
    text = re.sub(r"[\s.,:;]+\|", " |", text)
    # Thay khoảng trắng kép/tab thành khoảng trắng đơn
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def hard_masking(text):
    # Các từ này nếu xuất hiện là do cập nhật hồ sơ sau khi có kết quả
    # Ép thành [MASK] 100% để mô hình phải đọc triệu chứng (nôn ói, lơ mơ...)
    fatal_leakage_pattern = r"\b(i63|nhồi máu não|nhồi máu|nhồi máu cơ tim|xuất huyết não|xuất huyết|tai biến mạch máu não)\b"
    
    # Kể cả dấu ngoặc (i63) cũng bị xóa
    text = re.sub(r"\(i63\)", "[MASK]", text, flags=re.IGNORECASE)
    text = re.sub(fatal_leakage_pattern, "[MASK]", text, flags=re.IGNORECASE)
    
    return text

def dynamic_masking(text):
    # Từ "đột quỵ" hoặc "đột quỵ não" (chưa rõ thể loại nào)
    if random.random() < 0.5:
        soft_pattern = r"(chẩn đoán\s*)?(đột quỵ\s*n\~ao|đột quỵ|tai biến)"
        text = re.sub(soft_pattern, " [MASK] ", text, flags=re.IGNORECASE)
    return text

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

        # 3. Extract NIHSS score from all 5 original text columns combined
        combined_text_for_nihss = f"{lydo} {hb_benhly} {hb_banthan} {kb_toanthan} {kb_bophan}"
        nihss_match = nihss_pattern.search(combined_text_for_nihss)
        if nihss_match:
            score = nihss_match.group(1)
            nihss_str = f"Điểm đánh giá NIHSS là {score}."
        else:
            nihss_str = "Không ghi nhận đánh giá NIHSS."

        # Clean NIHSS info from columns before processing
        nihss_clean_pattern = re.compile(r'nihs{1,4}[:\s-]*\d+([\s]*điểm)?[\s.,-]*', re.IGNORECASE)
        lydo_clean = nihss_clean_pattern.sub("", lydo).strip()
        hb_benhly_clean = nihss_clean_pattern.sub("", hb_benhly).strip()
        hb_banthan_clean = nihss_clean_pattern.sub("", hb_banthan).strip()
        kb_toanthan_clean = nihss_clean_pattern.sub("", kb_toanthan).strip()
        kb_bophan_clean = nihss_clean_pattern.sub("", kb_bophan).strip()

        # Clean with advanced_clinical_text_cleaner
        lydo_clean = advanced_clinical_text_cleaner(lydo_clean)
        hb_benhly_clean = advanced_clinical_text_cleaner(hb_benhly_clean)
        hb_banthan_clean = advanced_clinical_text_cleaner(hb_banthan_clean)
        kb_toanthan_clean = advanced_clinical_text_cleaner(kb_toanthan_clean)
        kb_bophan_clean = advanced_clinical_text_cleaner(kb_bophan_clean)

        # 4. Concat clinical fields and apply dynamic masking
        clinical_text = (
            f"Lý do: {lydo_clean} | "
            f"Bệnh sử: {hb_benhly_clean} | "
            f"Tiền sử: {hb_banthan_clean} | "
            f"Toàn thân: {kb_toanthan_clean} | "
            f"Bộ phận: {kb_bophan_clean}"
        )
        clinical_text = hard_masking(clinical_text)
        clinical_text = dynamic_masking(clinical_text)

        # 5. Concat with NIHSS suffix
        input_text = f"{clinical_text} | {nihss_str}"
        # Final whitespace and punctuation clean-up
        input_text = re.sub(r"\s+", " ", input_text).strip()

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
