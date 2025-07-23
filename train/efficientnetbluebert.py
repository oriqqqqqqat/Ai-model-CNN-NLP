import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# เพิ่ม path เพื่อ import module ได้
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess_tensor.preprocessingblueBert import SkinDataset, get_tokenizer, get_transforms
from concatfeature.EfficienNetv2_BlueBert import MultimodalEfficientNetV2

# ===== CONFIG =====
train_csv = "D:/multimodalll/CNN+NLP/data/train.csv"
val_csv = "D:/multimodalll/CNN+NLP/data/val.csv"
batch_size = 8
num_epochs = 50
patience = 5
early_stop_threshold = 30  # ✅ หยุดหลังจาก epoch 30 ถ้า val_acc ไม่พัฒนา
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load CSVs =====
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

# ===== Label Encoding =====
class_names = sorted(train_df["disease"].unique().tolist())
label_map = {cls: i for i, cls in enumerate(class_names)}

# ===== Tokenizer & Transforms =====
tokenizer = get_tokenizer()
transform = get_transforms()

# ===== Dataset & Loader =====
train_dataset = SkinDataset(train_df, tokenizer, transform, label_map=label_map)
val_dataset = SkinDataset(val_df, tokenizer, transform, label_map=label_map)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ===== Model =====
model =  MultimodalEfficientNetV2(num_classes=len(class_names)).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

# ===== Evaluation Function =====
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images, input_ids, attention_mask)
            loss = torch.nn.functional.cross_entropy(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

# ===== Training Loop with Early Stopping =====
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0

    for batch in train_loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    val_loss, val_acc = evaluate(model, val_loader)

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"📊 Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ===== Save best model =====
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "newbest_modelefficientnet_bluebert.pth")
        print(f"💾 Saved BEST model (Val Acc: {val_acc:.4f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # ===== Save last model every epoch =====
    torch.save(model.state_dict(), "newlast_modelefficientnet_bluebert.pth")

    # ===== Modified Early Stopping Condition =====
    if epoch + 1 >= early_stop_threshold and epochs_no_improve >= patience:
        print(f"⏹️ Early stopping at epoch {epoch+1} "
              f"(no improvement in val_acc for {patience} epochs after epoch {early_stop_threshold})")
        break

# ===== Plotting =====
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(val_accuracies, label='Val Accuracy', color='green')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("efficientnetblueberttraining_plot.png")
plt.show()

print("✅ Training complete.")
