EXCEL_TEXT_COLUMNS = ["LYDO", "HB_BENHLY", "HB_BANTHAN", "HB_GIADINH", "KB_TOANTHAN", "KB_BOPHAN"]
EXCEL_MULTICLASS_LABELS = ["I63_INFARCTION", "OTHER_STROKE_LIKE", "DISTANT_OTHER"]
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
