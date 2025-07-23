from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image

def get_tokenizer():
    # ใช้ BlueBERT
    return AutoTokenizer.from_pretrained(
        "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
    )

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
        self.label_map = label_map  # ✅ รับ label_map จากภายนอก

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx] if hasattr(self.df, "iloc") else self.df[idx]

        # โหลดภาพจาก path
        try:
            image = Image.open(row["skincap_file_path"]).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"โหลดภาพไม่ได้: {row['skincap_file_path']} \n{e}")

        if self.transform:
            image = self.transform(image)

        # ใช้ caption เป็น input
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

        # ✅ แปลง label string เป็น index ด้วย label_map
        raw_label = row["disease"]
        if self.label_map:
            label = self.label_map[raw_label]
        else:
            label = raw_label  # fallback ถ้าไม่ใช้ label_map

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "path": row["skincap_file_path"] 
        }
