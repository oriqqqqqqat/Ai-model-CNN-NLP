import os
import pandas as pd
from datasets import load_dataset
from PIL import Image

# === CONFIG ===
meta_excel_path = "D:/multimodalll/CNN+NLP/data/skinimg.xlsx"
output_dir = "D:/multimodalll/CNN+NLP/data/skinimages_5diseases"

# โรคที่ต้องการ
target_diseases = [
    'basal-cell-carcinoma',
    'basal cell carcinoma',
    'squamous-cell-carcinoma',
    'squamous cell carcinoma',
    'melanocytic-nevi',
    'psoriasis',
    'lupus erythematosus'
]

# === 1. โหลด metadata ===
print("📋 Loading metadata...")
meta = pd.read_excel(meta_excel_path, header=1)

# === เลือกเฉพาะโรคที่ต้องการ ===
filtered_meta = meta[meta['disease'].isin(target_diseases)].reset_index(drop=True)
print(f"✅ Selected {len(filtered_meta)} records from metadata for target diseases.")

# === 2. โหลด dataset HuggingFace ===
print("📦 Loading image dataset from HuggingFace...")
dataset = load_dataset("joshuachou/SkinCAP", split="train")

# === 3. สร้างโฟลเดอร์สำหรับบันทึกภาพ ===
os.makedirs(output_dir, exist_ok=True)

# === 4. ดึงและเซฟภาพ ===
count = 0
for i, row in filtered_meta.iterrows():
    image_id = row['id']
    disease = row['disease']

    # แปลง image_id เป็น index ใน dataset HuggingFace
    dataset_index = int(image_id) - 1  # dataset เริ่มจาก 0

    # ตรวจสอบ index ให้อยู่ในช่วงของ dataset
    if dataset_index >= len(dataset) or dataset_index < 0:
        print(f"⚠️ Skipping: id {image_id} out of range.")
        continue

    # โหลดภาพจาก dataset
    img = dataset[dataset_index]["image"]

    # ตรวจสอบโหมดภาพ
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # ตั้งชื่อไฟล์
    disease_clean = disease.replace(" ", "_").replace("/", "_").replace("\\", "_")
    filename = f"{image_id}_{disease_clean}.jpg"
    filepath = os.path.join(output_dir, filename)

    # บันทึกภาพลง local
    img.save(filepath)
    count += 1
    print(f"✅ Saved: {filename}")

print(f"\n🎉 Done! Saved {count} images in '{output_dir}'")
