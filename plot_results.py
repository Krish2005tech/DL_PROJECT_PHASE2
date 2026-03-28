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

min_time = min(time)
relative_time = [t / min_time for t in time]

plt.figure()
plt.bar(models, relative_time)
plt.title("Training Time (Relative to Fastest)")
plt.ylabel("Relative Time")
plt.savefig("results/time_relative.png")


plt.figure()
plt.bar(models, time)
plt.ylim(min(time)*0.98, max(time)*1.02)  # adjust as needed
plt.title("Training Time Comparison (Zoomed)")
plt.savefig("results/time_zoomed.png")

plt.figure()
bars = plt.bar(models, time)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', 
             ha='center', va='bottom')

plt.title("Training Time Comparison")
plt.savefig("results/time_with_values.png")

plt.figure()
bars = plt.bar(models, time)

plt.ylim(min(time)*0.98, max(time)*1.02)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', 
             ha='center', va='bottom')

plt.title("Training Time Comparison (Zoomed)")
plt.savefig("results/time.png")