import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel

class MultimodalDenseNet(nn.Module):
    def __init__(self, num_classes=5):
        super(MultimodalDenseNet, self).__init__()

        # ใช้ DenseNet121 สำหรับภาพ
        self.vision_model = models.densenet121(weights='IMAGENET1K_V1')
        self.vision_model.classifier = nn.Identity()

        # โมเดลสำหรับข้อความ (BlueBERT)
        self.text_model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")

        # Fully connected layer สำหรับ classification
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 768, 256),  # 1024 สำหรับ DenseNet121 และ 768 สำหรับ BlueBERT
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        # ดึงฟีเจอร์จากภาพ
        vision_feat = self.vision_model(image)
        
        # ดึงฟีเจอร์จากข้อความ
        text_feat = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        
        # รวมฟีเจอร์จากภาพและข้อความ
        combined = torch.cat((vision_feat, text_feat), dim=1)
        
        # ผ่าน fully connected layer สำหรับการจำแนก
        return self.classifier(combined)
