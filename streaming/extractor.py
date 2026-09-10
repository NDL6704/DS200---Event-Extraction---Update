"""
extractor.py
Module chua lop EventExtractor thuc hien suy luan trich xuat thong tin su kien
(Trigger, Arguments, Event Type) su dung mo hinh OneIE JointModel (XLM-RoBERTa)
theo dung cau truc trong Pipeline_Joint_Learning.ipynb.
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import XLMRobertaTokenizerFast

# Thiet lap duong dan import module
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from models.model import JointModel
except ImportError:
    from model import JointModel


class EventExtractor:
    """
    Trich xuat su kien dua tren kien truc OneIE Joint Learning:
    - JointModel: XLM-RoBERTa voi 3 dau phan loai tuyen tinh (Trigger, Argument, Event Type).
    - Bo giai ma nhan: trigger_encoder.pkl (26 classes), arg_encoder.pkl (7 classes), event_encoder.pkl (14 classes).
    - Can chinh subword ve tu goc theo word_ids() cua tokenizer.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        encoder_dir: Optional[str] = None,
        model_name: str = "xlm-roberta-base",
        max_length: int = 128,
        device: Optional[str] = None,
    ):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_length = max_length
        self.model_name = model_name

        # 1. Tai Encoders
        if encoder_dir is None:
            encoder_dir = os.path.join(root_dir, "models", "encoders")
            if not os.path.exists(encoder_dir):
                encoder_dir = os.path.join(current_dir, "oneie_encoders")
        self.encoder_dir = Path(encoder_dir)

        print(f"[INFO] Dang tai cac bo ma hoa nhan OneIE tu: {self.encoder_dir}...")
        self.encoders = self._load_encoders(self.encoder_dir)

        self.num_trigger_labels = len(self.encoders["trigger"].classes_)
        self.num_arg_labels = len(self.encoders["arg"].classes_)
        self.num_event_labels = len(self.encoders["event"].classes_)
        print(
            f"[INFO] So lop nhan: Trigger = {self.num_trigger_labels} | "
            f"Argument = {self.num_arg_labels} | "
            f"Event Type = {self.num_event_labels}"
        )

        # 2. Khoi tao Tokenizer
        print(f"[INFO] Dang khoi tao Tokenizer '{model_name}'...")
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)

        # 3. Khoi tao va nap trong so JointModel
        if model_path is None:
            model_path = os.path.join(root_dir, "models", "checkpoints", "best_model.pt")
            if not os.path.exists(model_path):
                model_path = os.path.join(current_dir, "best_model.pt")

        self.model_path = model_path
        self.model = self._load_model(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"[INFO] OneIE EventExtractor da san sang tren thiet bi: {self.device}")

    def _load_encoders(self, encoder_dir: Path) -> Dict[str, Any]:
        encoders = {}
        for name in ["trigger", "arg", "event"]:
            pkl_path = encoder_dir / f"{name}_encoder.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(f"[ERROR] Khong tim thay file encoder: {pkl_path}")
            with open(pkl_path, "rb") as f:
                encoders[name] = pickle.load(f)
        return encoders

    def _load_model(self, model_path: str) -> JointModel:
        model = JointModel(
            self.model_name,
            self.num_trigger_labels,
            self.num_arg_labels,
            self.num_event_labels,
        )

        if os.path.exists(model_path):
            print(f"[INFO] Dang nap trong so OneIE JointModel tu: {model_path}...")
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print("[SUCCESS] Nap trong so OneIE JointModel thanh cong!")
            except Exception as e:
                print(f"[WARNING] Loi nap trong so ({e}).")
        else:
            print(f"[WARNING] Khong tim thay file checkpoint {model_path}.")

        return model

    def predict(self, text: str, article_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Trich xuat su kien va thuc the bang mo hinh OneIE Joint Learning.
        
        Args:
            text: Chuoi van ban can trich xuat.
            article_data: Du lieu bai bao goc (tuy chon).
            
        Returns:
            Dictionary theo dinh dang chuan:
            {
              "event_type": str,
              "entities": [
                {
                  "token": str,
                  "trigger_tag": str,
                  "argument_tag": str
                }, ...
              ]
            }
        """
        if not text or not text.strip():
            return {"event_type": "None", "entities": []}

        tokens = text.strip().split()
        if len(tokens) == 0:
            return {"event_type": "None", "entities": []}

        # 1. Tokenize voi word-level alignment (is_split_into_words=True)
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = encoding.word_ids()
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # 2. Lan truyen tien qua kien truc OneIE JointModel
        with torch.no_grad():
            trigger_logits, arg_logits, event_logits = self.model(input_ids, attention_mask)

        # 3. Lay nhan xac suat cao nhat (argmax)
        trigger_preds = torch.argmax(trigger_logits, dim=-1).cpu().numpy()[0]
        arg_preds = torch.argmax(arg_logits, dim=-1).cpu().numpy()[0]
        event_pred = torch.argmax(event_logits, dim=-1).cpu().item()

        # 4. Anh xa subword ve cac tu token goc
        predicted_tags = {}
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue
            if word_idx < len(tokens):
                predicted_tags[word_idx] = {
                    "trigger": self.encoders["trigger"].inverse_transform([trigger_preds[i]])[0],
                    "argument": self.encoders["arg"].inverse_transform([arg_preds[i]])[0],
                }
            previous_word_idx = word_idx

        # 5. Thu thap cac thuc the co nhan trigger hoac argument khac nhan 'O'
        extracted_entities = [
            {
                "token": tokens[i],
                "trigger_tag": predicted_tags[i]["trigger"],
                "argument_tag": predicted_tags[i]["argument"],
            }
            for i in range(len(tokens))
            if i in predicted_tags and (
                predicted_tags[i]["trigger"] != "O" or predicted_tags[i]["argument"] != "O"
            )
        ]

        predicted_event_type = self.encoders["event"].inverse_transform([event_pred])[0]

        return {
            "event_type": predicted_event_type,
            "entities": extracted_entities,
        }


if __name__ == "__main__":
    import json

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    extractor = EventExtractor()
    sample_text = (
        "Sang 1/7, tinh Hung Yen, Tinh uy, HDND, UBND, ban MTTQ Viet Nam tinh Hung Yen "
        "long trong to chuc Le ky niem 110 nam Ngay sinh Tong Bi thu Nguyen Van Cu."
    )
    res = extractor.predict(sample_text)
    print("\n--- TEST KET QUA TRICH XUAT ONEIE JOINT LEARNING ---")
    print(json.dumps(res, ensure_ascii=False, indent=2))
