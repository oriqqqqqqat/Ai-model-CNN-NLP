import pandas as pd
import glob

# หาไฟล์ .csv ทั้งหมดในโฟลเดอร์
csv_files = glob.glob('*.csv')

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    
    # ลบ ori_file_path ถ้ามี
    if 'ori_file_path' in df.columns:
        df = df.drop(columns=['ori_file_path'])
        print(f"✅ ลบคอลัมน์ ori_file_path จาก {csv_file} แล้ว")
    else:
        print(f"❌ {csv_file} ไม่มีคอลัมน์ ori_file_path")
    
    # แก้ชื่อ disease ถ้าต้องการเปลี่ยนชื่อ
    df['disease'] = df['disease'].replace({
        'basal-cell-carcinoma': 'basal cell carcinoma',
        'squamous-cell-carcinoma': 'squamous cell carcinoma'
        # เพิ่มโรคอื่นได้ถ้าต้องการ
    })

    df.to_csv(csv_file, index=False, encoding='utf-8-sig')

print("🎉 เสร็จสิ้น")
