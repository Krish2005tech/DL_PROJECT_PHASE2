import os
import torch
from torchvision import datasets

# Define the path
data_dir = './datasets/EuroSAT'
os.makedirs(data_dir, exist_ok=True)

print("🛰️ Downloading EuroSAT Dataset...")
# This will download and extract the dataset automatically
dataset = datasets.EuroSAT(root=data_dir, download=True)

print(f"✅ Download complete! Data is located in {data_dir}")
