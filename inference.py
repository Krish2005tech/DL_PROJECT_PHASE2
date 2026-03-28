import torch
import matplotlib.pyplot as plt
from dataset import DotaDataset
from models.quad_vim_model import QuadVimModel

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = DotaDataset("datasets/dota")

model = QuadVimModel().to(device)
model.eval()

img, _ = dataset[0]
img = img.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output)

plt.imshow(img[0].cpu().permute(1,2,0))
plt.title(f"Prediction: {pred.item()}")
plt.savefig("results/sample_output.png")
