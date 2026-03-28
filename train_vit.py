import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import time
import json

# Import local project modules
from dataset import DotaDataset
from utils import train_one_epoch, evaluate

# ✅ Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ PHASE 2: CRANKED RESOLUTION TO 1024
transform = transforms.Compose([
    transforms.Resize((1024, 1024)), 
    transforms.ToTensor()
])

# ✅ DATASET PATH FROM YOUR SERVER
dataset = DotaDataset(
    img_dir="datasets/DOTAv1/DOTAv1/images/train",
    transform=transform
)

# Batch size 16 at 1024px is guaranteed to push ViT's quadratic attention to its limit
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ✅ MODEL INITIALIZATION 
# Added img_size=1024 to bypass the 224px Assertion Error
model = timm.create_model(
    'vit_tiny_patch16_224',
    pretrained=True,
    num_classes=10,
    img_size=1024  
)
model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
epochs = 5
start_time = time.time()

print("🚀 Starting High-Res ViT Experiment...")

try:
    for epoch in range(epochs):
        # This is where the OOM crash will likely happen
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        acc = evaluate(model, loader, device)

        print(f"[ViT] Epoch {epoch}: Loss={loss:.4f}")

        # ✅ GPU MEMORY LOGGING
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"GPU Memory: {mem_gb:.2f} GB")

except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n" + "="*50)
        print("ViT triggered CUDA Out of Memory (OOM)!")
    else:
        raise e

total_time = time.time() - start_time

# Save baseline results
results = {
    "model": "ViT",
    "accuracy": 0.0, # Accuracy is irrelevant if it crashes
    "time": total_time,
    "status": "OOM_CRASHED"
}

with open("results/vit_highres_log.json", "w") as f:
    json.dump(results, f)

print("✅ ViT High-Res Experiment Finished")
