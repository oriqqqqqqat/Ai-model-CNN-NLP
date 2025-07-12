import pandas as pd
from sklearn.model_selection import train_test_split

# กำหนด path ไฟล์ input/output
meta_path = "skinimg.xlsx"
train_path = "train.csv"
val_path = "val.csv"
test_path = "test.csv"

# กำหนดรายชื่อโรคที่ต้องการ (case sensitive ตรงกับใน meta)
target_diseases = [
    'basal cell carcinoma',
    'basal-cell-carcinoma',
    'squamous cell carcinoma',
    'squamous-cell-carcinoma',
    'melanocytic-nevi',
    'psoriasis',
    'lupus erythematosus'
]

# อ่านไฟล์ metadata
df = pd.read_excel(meta_path, header=1)
df = df[df['disease'].isin(target_diseases)].reset_index(drop=True)

train_list, val_list, test_list = [], [], []

for disease in target_diseases:
    subset = df[df['disease'] == disease].sample(frac=1, random_state=42)  # shuffle
    
    # จำนวนภาพทั้งหมด
    total = len(subset)
    n_train = 67
    n_val = 14
    n_test = 14

    # กันกรณี class มีน้อยกว่าจำนวนที่ตั้งไว้ (เช่น data หาย)
    if total < (n_train + n_val + n_test):
        n_train = int(total * 0.7)
        n_val = int(total * 0.15)
        n_test = total - n_train - n_val

    train = subset.iloc[:n_train]
    val = subset.iloc[n_train:n_train + n_val]
    test = subset.iloc[n_train + n_val:n_train + n_val + n_test]
    
    train_list.append(train)
    val_list.append(val)
    test_list.append(test)

# รวมทุกโรคแต่ละชุด
train_df = pd.concat(train_list).reset_index(drop=True)
val_df = pd.concat(val_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

# shuffle อีกรอบ (เฉพาะในแต่ละชุด)
train_df = train_df.sample(frac=1, random_state=123).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=123).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=123).reset_index(drop=True)

# save
train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"✅ Done! แบ่งไฟล์ meta เป็น train/val/test เรียบร้อย")
print(f"จำนวนรูปในแต่ละชุด: train = {len(train_df)}, val = {len(val_df)}, test = {len(test_df)}")
