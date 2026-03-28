import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.vim import QuadVisionMamba
from datasets.dota_utils import DOTADataset

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LR = 1e-4

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Dataset
train_dataset = DOTADataset(
    img_dir='datasets/DOTAv1/DOTAv1/images/train',
    label_dir='datasets/DOTAv1/DOTAv1/labels/train',
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Model
model = QuadVisionMamba(num_classes=15).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print("🚀 Starting Quad-Vim Training on L40S...")

# Training loop
for epoch in range(5):
    model.train()

    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f}")

# ✅ ONLY ADDITION (your missing part)
torch.save(model.state_dict(), "model.pth")
print("✅ Model saved as model.pth")
