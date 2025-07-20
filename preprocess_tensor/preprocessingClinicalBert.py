from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
import torch

def get_tokenizer():
    return AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

class SkinDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None, label_map=None):
        self.df = df.reset_index(drop=True) if hasattr(df, "reset_index") else df
        self.tokenizer = tokenizer
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx] if hasattr(self.df, "iloc") else self.df[idx]

        # === Load image ===
        image_path = row["skincap_file_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"❌ โหลดภาพไม่ได้: {image_path} \n{e}")

        if self.transform:
            image = self.transform(image)

        # === Encode caption ===
        text = str(row["caption_en"]) if "caption_en" in row else str(row["caption"])
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # === Map label ===
        label_str = row["disease"]
        label = self.label_map[label_str] if self.label_map else label_str

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label),
            "path": image_path  # ✅ สำหรับ evaluate
        }
