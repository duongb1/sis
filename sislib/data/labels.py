EXCEL_TEXT_COLUMNS = ["LYDO", "HB_BENHLY", "HB_BANTHAN", "HB_GIADINH", "KB_TOANTHAN", "KB_BOPHAN"]
EXCEL_MULTICLASS_LABELS = ["I63_INFARCTION", "OTHER_STROKE_LIKE", "DISTANT_OTHER"]
MRI_STROKE_LIKE_MAJOR_ICD = {
    "I60",
    "I61",
    "I62",
    "I64",
    "I65",
    "I66",
    "I67",
    "I68",
    "I69",
    "G45",
    "G03",
    "G04",
    "G40",
    "H81",
    "R42",
    "R51",
    "S06",
    "D43",
    "Q28",
}
EXCEL_MULTICLASS_LABEL_MAP = {
    "I63_INFARCTION": "I63_INFARCTION",
    "OTHER_CEREBROVASCULAR": "OTHER_STROKE_LIKE",
    "STROKE_MIMIC_NEURO": "OTHER_STROKE_LIKE",
    "DISTANT_OTHER": "DISTANT_OTHER",
    "OTHER_STROKE_LIKE": "OTHER_STROKE_LIKE",
}
BINARY_I63_LABELS = ["khong", "co"]


def normalize_multiclass_label(label: str) -> str:
    label = "" if label is None else str(label).strip()
    return EXCEL_MULTICLASS_LABEL_MAP.get(label, label)


def binary_i63_from_multiclass(label: str) -> int:
    return 1 if normalize_multiclass_label(label) == "I63_INFARCTION" else 0


def major_icd_code(value) -> str:
    text = "" if value is None else str(value).strip().upper()
    if not text:
        return ""
    return text.split(".", 1)[0]


def mri_3class_from_maicd(value) -> str:
    major = major_icd_code(value)
    if major == "I63":
        return "I63_INFARCTION"
    if major in MRI_STROKE_LIKE_MAJOR_ICD:
        return "OTHER_STROKE_LIKE"
    return "DISTANT_OTHER"
