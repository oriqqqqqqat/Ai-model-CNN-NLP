import torch
import numpy as np
from PIL import Image
import gradio as gr
import sys
import os
import json
import random

# === Set Seed for Reproducibility ===
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# === Path Setting ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Import custom modules ===
from preprocess_tensor.preprocessingblueBert import get_tokenizer, get_transforms
from concatfeature.DensNet_BlueBert import MultimodalDenseNet

# === Load label_map ===
with open("label_map.json", "r") as f:
    label_map = json.load(f)
class_names = [label for label, _ in sorted(label_map.items(), key=lambda x: x[1])]

# === Load model ===
weights_path = "weights/newbest_modeldensenet_bluebert.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultimodalDenseNet(num_classes=len(class_names))
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

# === Tokenizer and transform ===
tokenizer = get_tokenizer()
transform = get_transforms()

# === Predict function with averaged softmax ===
def predict_single(image_path, symptom_text, num_runs=5):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Encode text
        encoded = tokenizer(symptom_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Predict multiple times for stability
        probs_total = np.zeros(len(class_names))
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(image_tensor, input_ids, attention_mask)
                probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                probs_total += probs
        probs_avg = probs_total / num_runs

        # Get prediction
        predicted_idx = int(np.argmax(probs_avg))
        predicted_label = class_names[predicted_idx]
        confidence = probs_avg[predicted_idx] * 100

        # Format result
        result = "🔍 **ผลการทำนาย:**\n\n"
        for i, prob in enumerate(probs_avg):
            result += f"- {class_names[i]}: **{prob * 100:.2f}%**\n"
        result += f"\n✅ ระบบวิเคราะห์ว่า: **{predicted_label}**\nความมั่นใจ: **{confidence:.2f}%**"

        # Return markdown + bar chart
        return result, {class_names[i]: float(probs_avg[i]) for i in range(len(class_names))}

    except Exception as e:
        return f"❌ เกิดข้อผิดพลาด: {str(e)}", None

# === Gradio UI ===
demo = gr.Interface(
    fn=predict_single,
    inputs=[
        gr.Image(type="filepath", label="📷 อัปโหลดภาพผิวหนัง"),
        gr.Textbox(lines=2, label="📝 กรุณากรอกอาการ (ภาษาไทยหรืออังกฤษ)")
    ],
    outputs=[
        gr.Markdown(),
        gr.Label(num_top_classes=5)
    ],
    title="🩺 ระบบวิเคราะห์โรคผิวหนังจากภาพ + อาการ",
    description="อัปโหลดภาพผิวหนัง และกรอกอาการเบื้องต้น แล้วระบบจะวิเคราะห์ว่าอาจเป็นโรคอะไร พร้อมความมั่นใจ (%)",
)

if __name__ == "__main__":
    demo.launch(share=True)
