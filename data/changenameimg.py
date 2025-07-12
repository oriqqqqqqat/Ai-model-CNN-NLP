import os
import glob

# ใส่ path โฟลเดอร์ train, validate, test
folders = ['train', 'validate', 'test']

for folder in folders:
    for file in glob.glob(os.path.join(folder, '*.jpg')):
        # แยกชื่อไฟล์เก่า เช่น 35_melanocytic-nevi.jpg
        base = os.path.basename(file)
        id_part = base.split('_')[0]   # เอาเฉพาะเลข id หน้า _
        new_name = f"{id_part}.jpg"
        new_path = os.path.join(folder, new_name)
        os.rename(file, new_path)
        print(f"{file} -> {new_path}")

print("✅ เปลี่ยนชื่อไฟล์เรียบร้อยแล้ว!")
