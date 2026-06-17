import re
import sys
import pandas as pd
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Directory settings
base_dir = Path("c:/Users/DUONG/Desktop/sis/data")
path_small = base_dir / "small.csv"
path_large = base_dir / "large.csv"

def clean_bullets_and_join(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Normalize Unicode to NFC first
    text = unicodedata.normalize('NFC', text)
    
    # 2. Split by newline to process line-by-line
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        
        # Remove leading bullet points: - + * •  and space
        line_str = re.sub(r'^[-+*•\uf000-\uf0ff\s]+', '', line_str)
        line_str = line_str.strip()
        
        if not line_str:
            continue
            
        # Ensure it ends with a comma if no punctuation is present
        if not line_str[-1] in '.,;:?!|':
            line_str += ','
            
        cleaned_lines.append(line_str)
        
    # Join lines with a single space
    joined_text = " ".join(cleaned_lines)
    
    # 3. Handle inline bullet points / dashes (e.g. "tốt - cổ mềm")
    joined_text = re.sub(r'(?:\s+[-+*•\uf000-\uf0ff]+\s*|\s*[-+*•\uf000-\uf0ff]+\s+)', ', ', joined_text)
    
    # 4. Clean up multiple punctuation marks (e.g. ", ," or ", ." or ".,")
    joined_text = re.sub(r',[\s,]+', ', ', joined_text)
    joined_text = re.sub(r',\s*([.;:?!|])', r'\1', joined_text)
    joined_text = re.sub(r'\s+', ' ', joined_text)
    
    # Strip leading/trailing space and final comma
    joined_text = joined_text.strip()
    if joined_text.endswith(','):
        joined_text = joined_text[:-1] + '.'
        
    return joined_text

def clinical_text_normalizer(text):
    if not isinstance(text, str):
        return ""
    
    # First, clean bullets and join lines
    text = clean_bullets_and_join(text)
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove parenthesized or bracketed ICD codes (e.g. (i63), [i63], (i63.9), (i25.5, e11, i50))
    # This protects GCS scores like g14, insulin doses like s15, and dates like t10/2021
    text = re.sub(r'\(\s*[a-z]\d{2}(?:\.\d{1,2})?[\*†‡\d]?(?:\s*[\s,;+]\s*[a-z]\d{2}(?:\.\d{1,2})?[\*†‡\d]?)*\s*\)', '', text)
    text = re.sub(r'\[\s*[a-z]\d{2}(?:\.\d{1,2})?[\*†‡\d]?(?:\s*[\s,;+]\s*[a-z]\d{2}(?:\.\d{1,2})?[\*†‡\d]?)*\s*\]', '', text)
    
    # 1. Spelling Corrections: Southern dialect "quị" -> "quỵ"
    text = re.sub(r"\bquị\b", "quỵ", text)
    
    # 2. Typos & Spelling Mistakes (New Round)
    text = re.sub(r"\bbenh\b", "bệnh", text)
    text = re.sub(r"\bnhan\b", "nhân", text)
    text = re.sub(r"\bkhong\b", "không", text)
    text = re.sub(r"\bđêu\b", "đều", text)
    text = re.sub(r"\bliêt\b", "liệt", text)
    text = re.sub(r"\bnhạp\b", "nhập", text)
    text = re.sub(r"\bnh[ậạ]p\s+v[ịệ]n\b", "nhập viện", text)
    text = re.sub(r"\bchuyên\s+viện\b", "chuyển viện", text)
    text = re.sub(r"\bbệnh\s+viện\s+tinh\b", "bệnh viện tỉnh", text)
    text = re.sub(r"\bbệnh\s+viện\s+huyên\b", "bệnh viện huyện", text)
    text = re.sub(r"\bxử\s+tri\b", "xử trí", text)
    text = re.sub(r"\bchuẩn\s+đoán\b", "chẩn đoán", text)
    text = re.sub(r"\bkhủy\b", "khuỷu", text)
    text = re.sub(r"\bmện\b", "mệt", text)
    text = re.sub(r"\bnhẹo\b", "nhẹ", text)
    text = re.sub(r"\blừ\s+đớ\b", "lừ đừ", text)
    text = re.sub(r"\bng\s+nhà\b", "người nhà", text)
    text = re.sub(r"\bcon\s+kéo\s+dài\b", "cơn kéo dài", text)
    text = re.sub(r"\b(\d+)\s*hút\b", r"\1 phút", text)
    text = re.sub(r"\btaijbv\b", "tại bệnh viện", text)
    text = re.sub(r"\btaij\b", "tại", text)
    text = re.sub(r"\b(thiếu\s+máu\s+cục\s+bộ|mạch\s+vành)\s+man\b", r"\1 mạn", text)
    
    # 2b. Accent & Accentless Typos (Round 2 Additions)
    text = re.sub(r"\bhuyêt\b", "huyết", text)
    text = re.sub(r"\bhuyết\s+ap\b", "huyết áp", text)
    text = re.sub(r"\bte\b", "tê", text)
    text = re.sub(r"\btrai\b", "trái", text)
    text = re.sub(r"\bdau\b", "đau", text)
    text = re.sub(r"\bphai\b", "phải", text)
    text = re.sub(r"\bnaoc\b", "não", text)
    text = re.sub(r"\b(xuất\s+huyết|nhồi\s+máu|u|phẫu\s+thuật)\s+nao\b", r"\1 não", text)
    text = re.sub(r"\bxo\s+kéo\b", "co kéo", text)
    text = re.sub(r"\bviệ\b", "viện", text)
    text = re.sub(r"\bnhâ\b", "nhân", text)
    text = re.sub(r"\byếu\s+nử\b", "yếu nửa", text)
    text = re.sub(r"\bnử\s+(người|bên|đầu|thân|phải|trái|mặt)\b", r"nửa \1", text)
    text = re.sub(r"\bhuyế\b", "huyết", text)
    
    # 2c. Round 3 Typos & Diacritics
    text = re.sub(r"\bđiều\s+tri\b", "điều trị", text)
    text = re.sub(r"\bđiều\s+trì\b", "điều trị", text)
    text = re.sub(r"\bxuât\b", "xuất", text)
    text = re.sub(r"\bgiât\b", "giật", text)
    text = re.sub(r"\ban\s+(uống|kém|ngủ|được|ngon)\b", r"ăn \1", text)
    text = re.sub(r"\bsức\s+co\b", "sức cơ", text)
    
    # 2d. Round 4 Typos & Diacritics
    text = re.sub(r"\bphục\s+phồi\b", "phục hồi", text)
    text = re.sub(r"\bphồi\b", "phổi", text)
    text = re.sub(r"\bkhó\s+thơ\b", "khó thở", text)
    text = re.sub(r"\bđột\s+quy\b", "đột quỵ", text)
    text = re.sub(r"\bxuát\b", "xuất", text)
    text = re.sub(r"\bnhip\b", "nhịp", text)
    text = re.sub(r"\bnaõ\b", "não", text)
    text = re.sub(r"\bbện\b", "bệnh", text)
    
    # 2e. Round 5 Typos & Diacritics (Edit Distance 1 & Units)
    text = re.sub(r"\b(cushinh|cuhing|cusing|cúshing)\b", "cushing", text)
    text = re.sub(r"\b(chơ|chỡ)\s+rẫy\b", "chợ rẫy", text)
    text = re.sub(r"\bchuyền\s+dạng\b", "chuyển dạng", text)
    text = re.sub(r"\bchuyền\b", "chuyển", text)
    text = re.sub(r"\bchuyến\b(?!\s+đi)", "chuyển", text)
    text = re.sub(r"\bđ\s+akhoa\b", "đa khoa", text)
    text = re.sub(r"\bakhoa\b", "đa khoa", text)
    text = re.sub(r"\bang\s+giang\b", "an giang", text)
    text = re.sub(r"\bbbên\b", "bên", text)
    text = re.sub(r"\bben\b", "bên", text)
    text = re.sub(r"\bbuốn\s+nôn\b", "buồn nôn", text)
    text = re.sub(r"\bbần\s+chân\b", "bàn chân", text)
    text = re.sub(r"\b(huyết|huyêt)\s+báp\b", r"\1 áp", text)
    text = re.sub(r"\bbâm\b", "bầm", text)
    text = re.sub(r"\bđau\s+bùng\b", "đau bụng", text)
    text = re.sub(r"\bquặn\s+bùng\b", "quặn bụng", text)
    text = re.sub(r"\bbùng\s+mềm\b", "bụng mềm", text)
    text = re.sub(r"\bbềm\b", "mềm", text)
    text = re.sub(r"\bđau\s+đầu\s+bền\b", "đau đầu bên", text)
    text = re.sub(r"\bbền\s+(phải|trái)\b", r"bên \1", text)
    text = re.sub(r"\bbệnhj\b", "bệnh", text)
    text = re.sub(r"\bhuyết\s+áp\s+cap\b", "huyết áp cao", text)
    text = re.sub(r"\bccho\b", "cho", text)
    text = re.sub(r"\bcháng\s+váng\b", "chóng váng", text)
    text = re.sub(r"\bchânn\b", "chân", text)
    text = re.sub(r"\bhoại\s+tử\s+chóm\b", "hoại tử chỏm", text)
    text = re.sub(r"\bchóm\s+xương\b", "chỏm xương", text)
    text = re.sub(r"\bchăn\s+đoán\b", "chẩn đoán", text)
    text = re.sub(r"\bchương\s+nhẹ\b", "chướng nhẹ", text)
    text = re.sub(r"\bchận\b", "chân", text)
    text = re.sub(r"\bchập\b", "chạp", text)
    text = re.sub(r"\bchằn\b", "chằng", text)
    text = re.sub(r"\bchna\b", "chân", text)
    text = re.sub(r"\bchép\b", "dép", text)
    text = re.sub(r"\bchụi\b", "chịu", text)
    text = re.sub(r"\bcoa\b", "có", text)
    text = re.sub(r"\bcàn\s+thơ\b", "cần thơ", text)
    text = re.sub(r"\bcám\s+giác\b", "cảm giác", text)
    text = re.sub(r"\bcáp\s+cứu\b", "cấp cứu", text)
    text = re.sub(r"\bcóng\s+mặt\b", "chóng mặt", text)
    text = re.sub(r"\bcđó\b", "đó", text)
    text = re.sub(r"\bcưo\b", "cơ", text)
    text = re.sub(r"\bcạch\b", "cách", text)
    text = re.sub(r"\bcảu\b", "của", text)
    text = re.sub(r"\bcúng\s+cơ\b", "cứng cơ", text)
    text = re.sub(r"\bmàng\s+cưng\b", "màng cứng", text)
    text = re.sub(r"\blang\s+cang\b", "lan can", text)
    text = re.sub(r"\bva\s+cham\b", "va chạm", text)
    text = re.sub(r"\bxấm\s+lấn\b", "xâm lấn", text)
    text = re.sub(r"\bnhiều\s+lấn\b", "nhiều lần", text)
    text = re.sub(r"\blấn\s+nữa\b", "lần nữa", text)
    text = re.sub(r"\s*l/ph\b", " lần/phút", text)
    
    # Context-aware replacement of 'măt' -> 'mặt' or 'mắt'
    text = re.sub(r"\bchóng\s+măt\b", "chóng mặt", text)
    text = re.sub(r"\btê\s+măt\b", "tê mặt", text)
    text = re.sub(r"\bliệt\s+măt\b", "liệt mặt", text)
    text = re.sub(r"\bvẻ\s+măt\b", "vẻ mặt", text)
    text = re.sub(r"\bmăt\s+nhắm\b", "mắt nhắm", text)
    text = re.sub(r"\bmăt\s+mở\b", "mắt mở", text)
    text = re.sub(r"\bmở\s+măt\b", "mở mắt", text)
    text = re.sub(r"\bmờ\s+măt\b", "mờ mắt", text)
    text = re.sub(r"\bnhắm\s+măt\b", "nhắm mắt", text)
    text = re.sub(r"\bsụp\s+mi\s+măt\b", "sụp mi mắt", text)
    text = re.sub(r"\bđộng\s+mạch\s+măt\b", "động mạch mắt", text)
    text = re.sub(r"\bxoay\s+măt\b", "xoay mắt", text)
    text = re.sub(r"\bmăt\s+(trái|phải)\b", r"mắt \1", text)
    text = re.sub(r"\bmăt\b", "mắt", text)
    
    # Context-aware replacement of 'mau' -> 'máu'
    text = re.sub(r"\b(nhồi|mạch|xuất\s+huyết|chảy|thiếu)\s+mau\b", r"\1 máu", text)
    
    # 3. Expand clinical abbreviations (Round 1 + Round 2)
    abbrev_map = {
        r"\bpxas\b": "phản xạ ánh sáng",
        r"\bnv\b": "nhập viện",
        r"\bha\b": "huyết áp",
        r"\btha\b": "tăng huyết áp",
        r"\bđtđ\b": "đái tháo đường",
        r"\btw\b": "trung ương",
        r"\bclvt\b": "cắt lớp vi tính",
        r"\bts\b": "tiền sử",
        r"\bxh\b": "xuất huyết",
        r"\bnmn\b": "nhồi máu não",
        r"\btsh\b": "tiêu sợi huyết",
        r"\bdktw\b": "đa khoa trung ương",
        r"\bcr\b": "chợ rẫy",
        r"\bbn\b": "bệnh nhân",
        r"\bbv\b": "bệnh viện",
        r"\bbvđk\b": "bệnh viện đa khoa",
        r"\b(dhmm|đhmm|đmmm)\b": "đường huyết mao mạch",
        r"\bđm\b": "động mạch",
        r"\bxq\b": "x-quang",
        r"\bnkq\b": "nội khí quản",
        r"\bphcn\b": "phục hồi chức năng",
        r"\btd\b": "theo dõi",
        r"\bbs\b": "bác sĩ",
        r"\bpxgx\b": "phản xạ gân xương",
        r"\bkhx\b": "kết hợp xương",
        r"\bvltl\b": "vật lý trị liệu",
        r"\bcstl\b": "cột sống thắt lưng",
        r"\btngt\b": "tai nạn giao thông",
        r"\bkv\b": "khu vực",
        r"\bbttmcb\b": "bệnh tim thiếu máu cục bộ",
        r"\brllpm\b": "rối loạn lipid máu"
    }
    for pattern, repl in abbrev_map.items():
        text = re.sub(pattern, repl, text)
        
    # 4. Expand clinical abbreviations (New Round 3)
    text = re.sub(r"\bttyt\b", "trung tâm y tế", text)
    text = re.sub(r"\bđktp\b", "đa khoa thành phố", text)
    text = re.sub(r"\bđhyd\b", "đại học y dược", text)
    text = re.sub(r"\bđk\b", "đa khoa", text)
    text = re.sub(r"\btx\b", "thị xã", text)
    
    # "tp" handling: "đái tháo đường tp 2" -> "đái tháo đường typ 2", others -> "thành phố"
    text = re.sub(r"\bđái\s+tháo\s+đường\s+tp\s+2\b", "đái tháo đường typ 2", text)
    text = re.sub(r"\btp\b", "thành phố", text)
    
    # "đt" handling: "bv đk đt" or similar -> "bệnh viện đa khoa đồng tháp"
    text = re.sub(r"\b(bệnh\s+viện|bv|đa\s+khoa|đk)\s+đt\b", r"\1 đồng tháp", text)
    
    # Syndromes & Clinical
    text = re.sub(r"\bhc\b", "hội chứng", text)
    text = re.sub(r"\btk\b", "thần kinh", text)
    text = re.sub(r"\bbc\s*:\s*(\d+)", r"bạch cầu: \1", text)
    text = re.sub(r"\bbc\s+(trái|phải)\b", r"bán cầu \1", text)
    text = re.sub(r"\b(bệnh\s+viện|bv)\s+tmct\b", r"\1 tim mạch cần thơ", text)
    text = re.sub(r"\btmct\b", "thiếu máu cơ tim", text)
    text = re.sub(r"\bvp\b(?!\s*shunt)", "viêm phổi", text)
    text = re.sub(r"\bvdd\b", "viêm dạ dày", text)
    text = re.sub(r"\bnt\s+(\d+)\b", r"nhịp thở \1", text)
    
    # Round 4 Additions
    text = re.sub(r"\bkt\b", "kích thước", text)
    text = re.sub(r"\bpk\b", "phòng khám", text)
    text = re.sub(r"\bpt\b", "phẫu thuật", text)
    text = re.sub(r"\bag\b", "an giang", text)
    text = re.sub(r"\btt\b", "trung tâm", text)
    text = re.sub(r"\b(bệnh\s+viện|bv)\s+tm\b", r"\1 tim mạch", text)
    text = re.sub(r"\bydtphcm\b", "y dược thành phố hồ chí minh", text)
    text = re.sub(r"\bđh\b", "đại học", text)
        
    # 5. Southern spelling "nữa" -> "nửa" (for body parts / sides)
    text = re.sub(r"\b(yếu|tê|liệt|đau)\s+nữa\b", r"\1 nửa", text)
    text = re.sub(r"\bnữa\s+(người|bên|đầu|thân|phải|trái|nguồi|ngươi|nửa|ngời|ngươì|ngươif|nhẹ|mặt)\b", r"nửa \1", text)
    text = re.sub(r"\b(nguồi|ngươi|ngời|ngươì|ngươif)\b", "người", text)
    text = re.sub(r"\bnửa\s+nửa\b", "nửa", text)
    
    # 6. Standardize direction markers p / t (parenthesized or isolated)
    # Match parenthesized (p) or (t) and replace with phải or trái
    text = re.sub(r"\s*\(\s*(p|phải)\s*\)", " phải", text)
    text = re.sub(r"\s*\(\s*(t|trái)\s*\)", " trái", text)
    
    # Match isolated p or t as independent words (e.g. "yếu 1/2 người p")
    text = re.sub(r"\b(p)\b", "phải", text)
    text = re.sub(r"\b(t)\b", "trái", text)
    
    # Clean up multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    # Clean up stray punctuation left by ICD removal or editing
    text = re.sub(r'\s+([.,;:?!])', r'\1', text) # "không ." -> "không."
    text = re.sub(r',[\s,]+', ', ', text)        # ", ," -> ", "
    text = re.sub(r';[\s;]+', '; ', text)        # "; ;" -> "; "
    text = re.sub(r'\.+', '.', text)             # "..." -> "."
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing space and final comma
    text = text.strip()
    if text.endswith(','):
        text = text[:-1] + '.'
        
    return text


def process_csv_file(path):
    if not path.exists():
        print(f"Error: {path.name} not found.")
        return
    
    df = pd.read_csv(path)
    print(f"\nProcessing CSV: {path.name} (shape: {df.shape})")
    
    text_cols = ['LYDO', 'HB_BENHLY', 'HB_BANTHAN', 'KB_TOANTHAN', 'KB_BOPHAN']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).apply(clinical_text_normalizer)
            
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"Successfully normalized and saved {path.name}.")

if __name__ == "__main__":
    process_csv_file(path_small)
    process_csv_file(path_large)
    print("\nCSV direct normalization completed successfully!")
