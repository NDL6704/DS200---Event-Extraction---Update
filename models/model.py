"""
model.py
Dinh nghia kien truc mo hinh OneIE Joint Learning (JointModel)
dua tren backbone XLM-RoBERTa ket hop 3 classifier heads tuyen tinh:
- Trigger Classifier (Token-level BIO Tagging)
- Argument Classifier (Token-level BIO Tagging)
- Event Classifier (Sequence-level Classification)
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaModel


class JointModel(nn.Module):
    """
    Mo hinh OneIE Joint Learning cho bai toan Event Extraction.
    Chia se bieu dien dac trung ngu nghia tu XLM-RoBERTa de dong thoi hoc 3 tac vu:
    1. Phan loai the BIO cho Trigger (tu kich hoat su kien).
    2. Phan loai the BIO cho Argument (cac thuc the tham gia).
    3. Phan loai loai su kien chinh tri (Event Type) o muc cau.
    """

    def __init__(
        self,
        model_name: str,
        num_trigger_labels: int,
        num_arg_labels: int,
        num_event_labels: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.xlm_roberta = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.xlm_roberta.config.hidden_size

        # Dau phan loai cho Trigger BIO tagging (Sequence output)
        self.trigger_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_trigger_labels),
        )

        # Dau phan loai cho Argument BIO tagging (Sequence output)
        self.arg_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_arg_labels),
        )

        # Dau phan loai cho Event Type classification (CLS token output)
        self.event_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_event_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]

        trigger_logits = self.trigger_classifier(sequence_output)
        arg_logits = self.arg_classifier(sequence_output)
        event_logits = self.event_classifier(cls_output)

        return trigger_logits, arg_logits, event_logits