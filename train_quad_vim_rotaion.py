import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import json

from dataset import DotaRotationDataset
from utils import train_one_epoch, evaluate
from models.quad_vim_model import QuadVimModel

# Setup
os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Rotation Dataset
dataset = DotaRotationDataset(img_dir="datasets/DOTAv1/DOTAv1/images/train", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize Quad-Vim with 4 classes
model = QuadVimModel(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

history = {"loss": [], "acc": []}
start_time = time.time()

print("🚀 Starting Rotation Training for Novelty: Quad-Vim...")

for epoch in range(10):
    loss = train_one_epoch(model, loader, optimizer, criterion, device)
    acc = evaluate(model, loader, device)
    
    history["loss"].append(loss)
    history["acc"].append(acc)
    
    print(f"[Quad-Vim Novelty] Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")

total_time = time.time() - start_time

# Save results for the final comparison graph
with open("results/quad_vim_rotation_results.json", "w") as f:
    json.dump({"history": history, "total_time": total_time}, f)

print("🔥 Quad-Vim Novelty Rotation Training Completed")
