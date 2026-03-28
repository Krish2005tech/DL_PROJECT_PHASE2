import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_eurosat_loaders(data_dir='./datasets/EuroSAT', batch_size=16):
    """
    EuroSAT loader with PROPER randomized split (no leakage)
    """

    # 🔒 Fix randomness for reproducibility
    torch.manual_seed(42)

    # ✅ Transform (correct)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ✅ Handle folder structure
    root_path = os.path.join(data_dir, '27500')
    if not os.path.exists(root_path):
        root_path = data_dir

    # ✅ Load dataset
    full_dataset = datasets.ImageFolder(
        root=root_path,
        transform=transform
    )

    # 🔥 CRITICAL FIX: shuffle BEFORE split
    indices = torch.randperm(len(full_dataset)).tolist()

    train_size = int(0.8 * len(full_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)

    # ✅ DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"✅ EuroSAT Ready: {len(train_ds)} train, {len(val_ds)} val images.")

    return train_loader, val_loader
