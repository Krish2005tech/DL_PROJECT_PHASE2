import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import json

from dataset import DotaDataset
from utils import train_one_epoch, evaluate
from models.vim_model import VimModel

# Create results folder
os.makedirs("results", exist_ok=True)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Transforms - 🔥 PHASE 2: CRANKED UP RESOLUTION TO 1024
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

# Dataset
dataset = DotaDataset(
    img_dir="datasets/DOTAv1/DOTAv1/images/train",
    transform=transform
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
model = VimModel().to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training
epochs = 5
start_time = time.time()

for epoch in range(epochs):
    loss = train_one_epoch(model, loader, optimizer, criterion, device)
    acc = evaluate(model, loader, device)

    # Modified: Removed Acc from print statement
    print(f"[Vim] Epoch {epoch}: Loss={loss:.4f}")

    # GPU memory logging (To prove it's lower than ViT)
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

total_time = time.time() - start_time

# Save results
results = {
    "model": "Vim",
    "accuracy": acc,
    "time": total_time
}

with open("results/vim_highres.json", "w") as f:
    json.dump(results, f)

print("✅ Vim High-Res Training Completed")
