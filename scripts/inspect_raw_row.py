import pandas as pd
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

pattern = r"(=*>|-*>|->)?\s*(?:(?:(nhập|chuyển)\s*(viện\s*)?(bệnh viện\s*)?(s\.i\.s|sis)(\s*cần thơ)?)|nv|nhập\s*viện)[\s.,]*$"

def search_exact(filename):
    df = pd.read_excel(filename)
    for col in df.columns:
        matched = df[df[col].astype(str).str.contains('nghẽn tắc động mạch đốt sống', case=False, na=False)]
        if not matched.empty:
            print(f"File {filename}, Col {col}:")
            for idx, val in zip(matched.index, matched[col]):
                print(f"  Row {idx}:")
                print(f"  Raw: {repr(val)}")
                text = str(val).lower()
                cleaned = re.sub(pattern, "", text)
                print(f"  Cleaned: {repr(cleaned)}")

search_exact('9937_co.xlsx')
search_exact('9937_khong.xlsx')
