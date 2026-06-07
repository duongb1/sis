import re


TEXT_FIELDS_ALL = ["LYDO", "HB_BENHLY", "HB_BANTHAN", "HB_GIADINH", "KB_TOANTHAN", "KB_BOPHAN"]
TEXT_FIELDS_CHIEF_EXAM = ["LYDO", "HB_BENHLY", "KB_TOANTHAN", "KB_BOPHAN"]

CLINICAL_CONCEPTS = [
    {
        "name": "motor_weakness",
        "group": "neuro_symptom",
        "patterns": [
            r"\bliệt\b",
            r"liệt\s*(nửa|1/2)",
            r"yếu\s*(nửa|1/2|tay|chân)",
            r"yếu\s*\d*/\d*",
            r"yếu.*tay",
            r"yếu.*chân",
            r"hemiparesis",
            r"hemiplegia",
        ],
    },
    {
        "name": "speech_problem",
        "group": "neuro_symptom",
        "patterns": [
            r"nói\s*(khó|đớ|ngọng|không rõ)",
            r"khó\s*nói",
            r"mất\s*ngôn\s*ngữ",
            r"aphasia",
            r"dysarthria",
        ],
    },
    {
        "name": "facial_palsy",
        "group": "neuro_symptom",
        "patterns": [
            r"méo\s*miệng",
            r"liệt\s*(mặt|vii|7)",
            r"liệt\s*dây\s*thần\s*kinh\s*(vii|7)",
        ],
    },
    {
        "name": "sensory_numbness",
        "group": "neuro_symptom",
        "patterns": [
            r"\btê\b",
            r"tê\s*(tay|chân|nửa|1/2)",
            r"dị\s*cảm",
            r"giảm\s*cảm\s*giác",
        ],
    },
    {
        "name": "dizziness_vertigo",
        "group": "neuro_symptom",
        "patterns": [
            r"chóng\s*mặt",
            r"choáng\s*váng",
            r"xây\s*xẩm",
            r"vertigo",
        ],
    },
    {"name": "headache", "group": "neuro_symptom", "patterns": [r"đau\s*đầu", r"nhức\s*đầu"]},
    {
        "name": "seizure",
        "group": "neuro_mimic",
        "patterns": [r"co\s*giật", r"động\s*kinh", r"seizure", r"epilep"],
    },
    {
        "name": "altered_consciousness",
        "group": "neuro_symptom",
        "patterns": [
            r"lơ\s*mơ",
            r"hôn\s*mê",
            r"rối\s*loạn\s*tri\s*giác",
            r"mất\s*ý\s*thức",
            r"glasgow",
            r"\bgcs\b",
        ],
    },
    {
        "name": "hypertension",
        "group": "risk_factor",
        "patterns": [r"tăng\s*huyết\s*áp", r"\btha\b", r"hypertension"],
    },
    {
        "name": "diabetes",
        "group": "risk_factor",
        "patterns": [r"đái\s*tháo\s*đường", r"tiểu\s*đường", r"\bdm\b", r"diabetes"],
    },
    {
        "name": "cardiac_risk",
        "group": "risk_factor",
        "patterns": [
            r"rung\s*nhĩ",
            r"bệnh\s*mạch\s*vành",
            r"suy\s*tim",
            r"nhồi\s*máu\s*cơ\s*tim",
            r"stent",
            r"van\s*tim",
            r"atrial\s*fibrillation",
        ],
    },
    {
        "name": "dyslipidemia",
        "group": "risk_factor",
        "patterns": [r"rối\s*loạn\s*lipid", r"tăng\s*mỡ\s*máu", r"tăng\s*cholesterol", r"dyslipidemia"],
    },
    {
        "name": "ataxia_balance",
        "group": "neuro_symptom",
        "patterns": [r"đi\s*loạng\s*choạng", r"mất\s*thăng\s*bằng", r"thất\s*điều", r"ataxia"],
    },
]

CONCEPT_GROUPS = ["neuro_symptom", "neuro_mimic", "risk_factor"]
CONCEPT_TO_GROUP = {concept["name"]: concept["group"] for concept in CLINICAL_CONCEPTS}
CONCEPT_NAMES = [concept["name"] for concept in CLINICAL_CONCEPTS]

_COMPILED_CONCEPTS = [
    {
        "name": concept["name"],
        "group": concept["group"],
        "patterns": [re.compile(pattern, flags=re.IGNORECASE) for pattern in concept["patterns"]],
    }
    for concept in CLINICAL_CONCEPTS
]


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(s):
            return ""
    except Exception:
        pass
    text = str(s).lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_concepts_by_field(row: dict, fields: list[str]) -> dict:
    concepts_by_field = {}
    for field in fields:
        text = normalize_text(row.get(field, ""))
        found = set()
        if text:
            for concept in _COMPILED_CONCEPTS:
                if any(pattern.search(text) for pattern in concept["patterns"]):
                    found.add(concept["name"])
        concepts_by_field[field] = found
    return concepts_by_field


def build_text(row: dict, fields: list[str]) -> str:
    parts = []
    for field in fields:
        value = normalize_text(row.get(field, ""))
        if value:
            parts.append(f"[{field}] {value}")
    return " ".join(parts)
