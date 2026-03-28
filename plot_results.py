import json
import matplotlib.pyplot as plt

with open("results/vit.json") as f:
    vit = json.load(f)

with open("results/vim.json") as f:
    vim = json.load(f)

with open("results/quad_vim.json") as f:
    quad = json.load(f)

models = ["ViT", "Vim", "Quad-Vim"]
accuracy = [vit["accuracy"], vim["accuracy"], quad["accuracy"]]
time = [vit["time"], vim["time"], quad["time"]]

plt.figure()
plt.bar(models, accuracy)
plt.title("Accuracy Comparison")
plt.savefig("results/accuracy.png")

plt.figure()
plt.bar(models, time)
plt.title("Training Time Comparison")
plt.savefig("results/time.png")
