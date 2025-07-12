import os
import shutil
import pandas as pd

# Path
src_img_dir = "D:/multimodalll/CNN+NLP/data/skinimages_5diseases"  # โฟลเดอร์ภาพต้นทาง
dst_root = "D:/multimodalll/CNN+NLP/data/split"                    # โฟลเดอร์ปลายทาง train/val/test
os.makedirs(dst_root, exist_ok=True)

# วนแต่ละชุด
for set_name in ['train', 'val', 'test']:
    df = pd.read_csv(f"{set_name}.csv")
    for i, row in df.iterrows():
        image_id = row['id']
        disease = row['disease']
        # สร้างชื่อไฟล์
        disease_clean = disease.replace(' ', '_').replace('/', '_')
        filename = f"{image_id}_{disease_clean}.jpg"

        # ต้นทาง
        src = os.path.join(src_img_dir, filename)
        # ปลายทาง (แยก folder ตาม label ด้วย)
        dst_dir = os.path.join(dst_root, set_name, disease_clean)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, filename)

        # คัดลอก
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"{filename} -> {set_name}/{disease_clean}")
        else:
            print(f"⚠️ Not found: {filename}")

print("✅ แยกรูปภาพตาม train/val/test และ label เรียบร้อยแล้ว!")
