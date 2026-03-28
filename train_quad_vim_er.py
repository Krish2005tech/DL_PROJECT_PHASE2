import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from dataset import get_eurosat_loaders
from models.quad_vim_model import QuadVimModel
from utils import train_one_epoch, evaluate

# Setup
os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load EuroSAT (10 classes)
train_loader, val_loader = get_eurosat_loaders(batch_size=16)

# Initialize Quad-Vim - Ensure num_classes=10
model = QuadVimModel(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

history = {"loss": [], "acc": []}
start_time = time.time()

print("🚀 Starting EuroSAT Training: Quad-Vim Novelty...")

for epoch in range(10):
    loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    acc = evaluate(model, val_loader, device)
    
    history["loss"].append(loss)
    history["acc"].append(acc)
    
    print(f"[Quad-Vim EuroSAT] Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")

total_time = time.time() - start_time

# Save results for comparison
with open("results/quad_vim_eurosat_results.json", "w") as f:
    json.dump({"history": history, "total_time": total_time}, f)

print("🔥 Quad-Vim Novelty EuroSAT Training Completed")
