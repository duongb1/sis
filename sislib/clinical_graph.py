import torch

from .clinical_concepts import CONCEPT_GROUPS, CONCEPT_NAMES, CONCEPT_TO_GROUP, extract_concepts_by_field


NODE_TYPES = {
    "patient": 0,
    "field": 1,
    "concept": 2,
    "group": 3,
}
EDGE_TYPES = {
    "has_field": 0,
    "mentions": 1,
    "belongs_to": 2,
    "related_to": 3,
    "self_loop": 4,
}
RELATED_CONCEPT_EDGES = [
    ("motor_weakness", "facial_palsy"),
    ("motor_weakness", "speech_problem"),
    ("motor_weakness", "sensory_numbness"),
    ("speech_problem", "facial_palsy"),
    ("dizziness_vertigo", "ataxia_balance"),
    ("seizure", "altered_consciousness"),
    ("hypertension", "headache"),
    ("diabetes", "cardiac_risk"),
    ("dyslipidemia", "cardiac_risk"),
]


def require_pyg():
    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError("PyTorch Geometric is required. Install it with: pip install torch_geometric") from exc
    return Data


def node_name_vocab(fields):
    names = ["patient"]
    names.extend(fields)
    names.extend(CONCEPT_NAMES)
    names.extend(CONCEPT_GROUPS)
    return {name: index for index, name in enumerate(dict.fromkeys(names))}


def add_bidirectional_edge(edges, edge_types, src, dst, edge_type):
    edges.append((src, dst))
    edge_types.append(EDGE_TYPES[edge_type])
    edges.append((dst, src))
    edge_types.append(EDGE_TYPES[edge_type])


def build_clinical_graph(row, fields, label, node_name_to_id=None):
    Data = require_pyg()
    node_name_to_id = node_name_to_id or node_name_vocab(fields)
    concepts_by_field = extract_concepts_by_field(row, fields)
    detected_concepts = sorted(set().union(*concepts_by_field.values())) if concepts_by_field else []
    detected_groups = sorted({CONCEPT_TO_GROUP[concept] for concept in detected_concepts})

    nodes = [("patient", "patient")]
    nodes.extend(("field", field) for field in fields)
    nodes.extend(("concept", concept) for concept in detected_concepts)
    nodes.extend(("group", group) for group in CONCEPT_GROUPS)

    node_index = {(node_type, name): index for index, (node_type, name) in enumerate(nodes)}
    edges, edge_types = [], []
    patient_idx = node_index[("patient", "patient")]

    for field in fields:
        field_idx = node_index[("field", field)]
        add_bidirectional_edge(edges, edge_types, patient_idx, field_idx, "has_field")
        for concept in sorted(concepts_by_field.get(field, set())):
            concept_idx = node_index[("concept", concept)]
            add_bidirectional_edge(edges, edge_types, field_idx, concept_idx, "mentions")

    for concept in detected_concepts:
        concept_idx = node_index[("concept", concept)]
        group_idx = node_index[("group", CONCEPT_TO_GROUP[concept])]
        add_bidirectional_edge(edges, edge_types, concept_idx, group_idx, "belongs_to")

    detected_set = set(detected_concepts)
    for left, right in RELATED_CONCEPT_EDGES:
        if left in detected_set and right in detected_set:
            add_bidirectional_edge(
                edges,
                edge_types,
                node_index[("concept", left)],
                node_index[("concept", right)],
                "related_to",
            )

    for index in range(len(nodes)):
        edges.append((index, index))
        edge_types.append(EDGE_TYPES["self_loop"])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_type_id = torch.tensor([NODE_TYPES[node_type] for node_type, _ in nodes], dtype=torch.long)
    node_name_id = torch.tensor([node_name_to_id[name] for _, name in nodes], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        node_type_id=node_type_id,
        node_name_id=node_name_id,
        y=torch.tensor([float(label)], dtype=torch.float32),
    )
    data.detected_concepts = detected_concepts
    return data
