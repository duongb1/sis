import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DualStreamPhoBERTMRIAlign(nn.Module):
    def __init__(self, model_name, num_labels=2, aux_dim=256, mri_dim=2048, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.main_head = nn.Sequential(
            nn.Linear(hidden_size, aux_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(aux_dim),
        )
        self.classifier = nn.Linear(aux_dim, num_labels)
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_size, aux_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(aux_dim),
            nn.Linear(aux_dim, mri_dim),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
        teacher_mri_vec=None,
        lambda_align=0.05,
        align_loss="cosine",
        detach_aux=False,
    ):
        encoder_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_inputs)
        h_cls = outputs.last_hidden_state[:, 0, :]

        h_main = self.main_head(h_cls)
        logits = self.classifier(h_main)

        aux_input = h_cls.detach() if detach_aux else h_cls
        pred_mri_vec = self.aux_head(aux_input)

        loss = loss_ce = loss_align = None
        if labels is not None:
            loss_ce = F.cross_entropy(logits, labels)
            loss = loss_ce

        if teacher_mri_vec is not None:
            if teacher_mri_vec.dim() == 3 and teacher_mri_vec.size(1) == 1:
                teacher_mri_vec = teacher_mri_vec.squeeze(1)
            teacher_mri_vec = teacher_mri_vec.float()
            pred_mri_vec = pred_mri_vec.float()
            if pred_mri_vec.shape != teacher_mri_vec.shape:
                raise ValueError(f"Shape mismatch: pred_mri_vec={pred_mri_vec.shape}, teacher_mri_vec={teacher_mri_vec.shape}")

            if align_loss == "cosine":
                pred_norm = F.normalize(pred_mri_vec, p=2, dim=-1)
                teacher_norm = F.normalize(teacher_mri_vec, p=2, dim=-1)
                loss_align = 1.0 - F.cosine_similarity(pred_norm, teacher_norm, dim=-1).mean()
            elif align_loss == "mse":
                loss_align = F.mse_loss(pred_mri_vec, teacher_mri_vec)
            else:
                raise ValueError(f"Unsupported align_loss: {align_loss}")

            loss = lambda_align * loss_align if loss is None else loss + lambda_align * loss_align

        return {
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_align": loss_align,
            "logits": logits,
            "h_main": h_main,
            "pred_mri_vec": pred_mri_vec,
        }
