import pandas as pd

# ตั้งค่า path ของไฟล์
files = [
    ('train.csv', r'D:\multimodalll\CNN+NLP\data\train'),
    ('val.csv',   r'D:\multimodalll\CNN+NLP\data\validate'),
    ('test.csv',  r'D:\multimodalll\CNN+NLP\data\test')
]

for csv_file, folder_path in files:
    df = pd.read_csv(csv_file)
    # สร้าง path เต็มใหม่
    df['skincap_file_path'] = df['id'].astype(str) + '.jpg'
    df['skincap_file_path'] = folder_path + '\\' + df['skincap_file_path']
    # Save ทับไฟล์เดิม (หรือตั้งชื่อใหม่ก็ได้)
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✅ อัปเดต skincap_file_path ใน {csv_file} แล้ว")

print("🎉 เสร็จสิ้น")
