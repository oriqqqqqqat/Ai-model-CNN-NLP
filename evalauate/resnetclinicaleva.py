import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)


# เพิ่ม path เพื่อ import module ได้
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess_tensor.preprocessingClinicalBert import SkinDataset, get_tokenizer, get_transforms
from concatfeature.ResNet50_clinicalbert import MultimodalResNet50


# ===== CONFIG =====
test_csv = "D:/multimodalll/CNN+NLP/data/test.csv"
model_path = "D:/multimodalll/CNN+NLP/weights/newbest_modelresnet_clinicalbert.pth"
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load Test Data =====
test_df = pd.read_csv(test_csv)
class_names = sorted(test_df["disease"].unique().tolist())
label_map = {cls: i for i, cls in enumerate(class_names)}
reverse_label_map = {v: k for k, v in label_map.items()}

tokenizer = get_tokenizer()
transform = get_transforms()

test_dataset = SkinDataset(test_df, tokenizer, transform, label_map=label_map)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===== Load Model =====
model = MultimodalResNet50(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===== Collect Predictions =====
y_true, y_pred = [], []
file_paths = []

with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images, input_ids, attention_mask)
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        file_paths.extend(batch["path"])

# ===== Confusion Matrix =====
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

plt.figure(figsize=(10, 8))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_resnetclinicalbert.png")
plt.show()
print("✅ Saved confusion_matrix_resnetclinicalbert.png")

# ===== Show and Save Predictions (Correct / Incorrect) =====
def show_and_save_predictions(file_paths, y_true, y_pred, reverse_map, correct=True,
                              save_path="summary.jpg", max_images=6, title=""):
    plt.figure(figsize=(12, 6))
    shown = 0
    for i in range(len(y_true)):
        is_correct = y_true[i] == y_pred[i]
        if is_correct != correct:
            continue
        try:
            img = Image.open(file_paths[i]).convert("RGB")
            plt.subplot(2, 3, shown + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"True: {reverse_map[y_true[i]]}\nPred: {reverse_map[y_pred[i]]}",
                      color="green" if is_correct else "red")
            shown += 1
            if shown >= max_images:
                break
        except Exception as e:
            print(f"❌ Load image failed: {file_paths[i]}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Saved and displayed: {save_path}")

# ✅ แสดง + เซฟภาพทำนายถูก
show_and_save_predictions(file_paths, y_true, y_pred, reverse_label_map,
                          correct=True,
                          save_path="correct_predictionsresnetclinicalbert_summary.jpg",
                          title="Predictions Correct")

# ❌ แสดง + เซฟภาพทำนายผิด
show_and_save_predictions(file_paths, y_true, y_pred, reverse_label_map,
                          correct=False,
                          save_path="incorrect_predictionsresnetclinicalbert_summary.jpg",
                          title="Predictions Incorrect")

# ===== Evaluate Performance Metrics =====
acc = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

report = classification_report(y_true, y_pred, target_names=class_names)

# ===== Save metrics to text file =====
with open("model_performance_resnetclinicalbert.txt", "w", encoding="utf-8") as f:
    f.write("📊 Model Performance Metrics\n")
    f.write("===========================\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")

    f.write("Macro Average:\n")
    f.write(f"  Precision: {precision_macro:.4f}\n")
    f.write(f"  Recall:    {recall_macro:.4f}\n")
    f.write(f"  F1 Score:  {f1_macro:.4f}\n\n")

    f.write("Weighted Average:\n")
    f.write(f"  Precision: {precision_weighted:.4f}\n")
    f.write(f"  Recall:    {recall_weighted:.4f}\n")
    f.write(f"  F1 Score:  {f1_weighted:.4f}\n\n")

    f.write("Full Classification Report:\n")
    f.write(report)

print("✅ Saved model_performance_resnetclinicalbert.txt")
