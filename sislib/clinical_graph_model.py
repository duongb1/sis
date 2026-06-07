import torch
import torch.nn as nn
import torch.nn.functional as F

from .clinical_graph import NODE_TYPES
from .text_train import AttentionPooling


def require_pyg_nn(graph_conv):
    try:
        from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
    except ImportError as exc:
        raise ImportError("PyTorch Geometric is required. Install it with: pip install torch_geometric") from exc
    convs = {"gat": GATConv, "gcn": GCNConv}
    if graph_conv not in convs:
        raise ValueError(f"Unsupported graph_conv: {graph_conv}")
    return convs[graph_conv], global_mean_pool


class ClinicalGraphEncoder(nn.Module):
    def __init__(
        self,
        num_node_names,
        graph_hidden_dim=64,
        graph_layers=2,
        graph_dropout=0.2,
        graph_heads=2,
        graph_conv="gat",
        graph_pooling="patient",
    ):
        super().__init__()
        if graph_pooling not in {"patient", "mean"}:
            raise ValueError(f"Unsupported graph_pooling: {graph_pooling}")
        conv_cls, global_mean_pool = require_pyg_nn(graph_conv)
        self.global_mean_pool = global_mean_pool
        self.graph_pooling = graph_pooling
        self.graph_conv = graph_conv
        self.node_type_embedding = nn.Embedding(len(NODE_TYPES), graph_hidden_dim)
        self.node_name_embedding = nn.Embedding(num_node_names, graph_hidden_dim)
        self.dropout = nn.Dropout(graph_dropout)
        self.convs = nn.ModuleList()
        for _ in range(graph_layers):
            if graph_conv == "gat":
                self.convs.append(conv_cls(graph_hidden_dim, graph_hidden_dim, heads=graph_heads, concat=False))
            else:
                self.convs.append(conv_cls(graph_hidden_dim, graph_hidden_dim))

    def forward(self, graph_data):
        h = self.node_type_embedding(graph_data.node_type_id) + self.node_name_embedding(graph_data.node_name_id)
        for conv in self.convs:
            h = conv(h, graph_data.edge_index)
            h = F.relu(h)
            h = self.dropout(h)
        if self.graph_pooling == "mean":
            return self.global_mean_pool(h, graph_data.batch)
        patient_mask = graph_data.node_type_id == NODE_TYPES["patient"]
        z_graph = h[patient_mask]
        batch_size = int(graph_data.num_graphs)
        if z_graph.size(0) != batch_size:
            raise RuntimeError(f"Expected one patient node per graph, got patient_nodes={z_graph.size(0)} batch_size={batch_size}")
        return z_graph


class ClinicalGraphOnlyClassifier(nn.Module):
    def __init__(self, num_node_names, graph_hidden_dim=64, graph_layers=2, graph_dropout=0.2, graph_heads=2, graph_conv="gat", graph_pooling="patient"):
        super().__init__()
        self.graph_encoder = ClinicalGraphEncoder(
            num_node_names,
            graph_hidden_dim=graph_hidden_dim,
            graph_layers=graph_layers,
            graph_dropout=graph_dropout,
            graph_heads=graph_heads,
            graph_conv=graph_conv,
            graph_pooling=graph_pooling,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(graph_dropout),
            nn.Linear(graph_hidden_dim, 1),
        )

    def forward(self, graphs, return_debug=False):
        z_graph = self.graph_encoder(graphs)
        logits = self.classifier(z_graph).squeeze(-1)
        if return_debug:
            return {"logits": logits, "z_graph": z_graph}
        return logits


class PhoBERTClinicalGraphFusion(nn.Module):
    base_model_prefix = "encoder"

    def __init__(
        self,
        model_name,
        num_node_names,
        graph_hidden_dim=64,
        graph_layers=2,
        graph_dropout=0.2,
        graph_heads=2,
        graph_conv="gat",
        graph_pooling="patient",
        dropout=0.1,
        pooling="attention",
    ):
        super().__init__()
        from transformers import AutoModel

        if pooling not in {"cls", "attention"}:
            raise ValueError(f"Unsupported text pooling for fusion: {pooling}")
        self.encoder = AutoModel.from_pretrained(model_name)
        text_hidden_size = self.encoder.config.hidden_size
        self.pooling = pooling
        self.attn_pool = AttentionPooling(text_hidden_size, dropout=dropout) if pooling == "attention" else None
        self.graph_encoder = ClinicalGraphEncoder(
            num_node_names,
            graph_hidden_dim=graph_hidden_dim,
            graph_layers=graph_layers,
            graph_dropout=graph_dropout,
            graph_heads=graph_heads,
            graph_conv=graph_conv,
            graph_pooling=graph_pooling,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_size + graph_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.config = self.encoder.config

    def encode_text(self, input_ids, attention_mask, token_type_ids=None):
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**model_inputs)
        hidden = outputs.last_hidden_state
        if self.pooling == "attention":
            pooled, _ = self.attn_pool(hidden, attention_mask)
        else:
            pooled = hidden[:, 0]
        return pooled

    def forward(self, input_ids, attention_mask, graphs, token_type_ids=None, return_debug=False):
        z_text = self.encode_text(input_ids, attention_mask, token_type_ids=token_type_ids)
        z_graph = self.graph_encoder(graphs)
        if z_text.size(0) != z_graph.size(0):
            raise RuntimeError(f"Text/graph batch mismatch: z_text={tuple(z_text.shape)} z_graph={tuple(z_graph.shape)}")
        logits = self.classifier(self.dropout(torch.cat([z_text, z_graph], dim=-1))).squeeze(-1)
        if return_debug:
            return {"logits": logits, "z_text": z_text, "z_graph": z_graph}
        return logits
