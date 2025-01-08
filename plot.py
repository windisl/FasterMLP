import matplotlib.pyplot as plt
import itertools

# Data preparation
models = {
    "ShuffleNetV2": {"Latency": 12.1, "Top-1": 72.6},
    "MobileNetV2": {"Latency": 12.2, "Top-1": 72.0},
    "EdgeNeXt": {"Latency": 15.5, "Top-1": 71.2},
    "EfficientFormer-L1": {"Latency": 14.3, "Top-1": 73.4},
    "MobileViT-XXS": {"Latency": 30.8, "Top-1": 69.0},
    "FasterMLP-S (Ours)": {"Latency": 11.38, "Top-1": 72.9},
    "ConvNeXt-T": {"Latency": 86.3, "Top-1": 82.1},
    "Swin-T": {"Latency": 122.2, "Top-1": 81.3},
    "Poolformer": {"Latency": 138.0, "Top-1": 81.4},
    "EfficientFormer-L3": {"Latency": 32.7, "Top-1": 79.2},
    "FasterMLP-L (Ours)": {"Latency": 94.3, "Top-1": 81.5},
}

# Extract data for plotting
latency = [v["Latency"] for v in models.values()]
top1 = [v["Top-1"] for v in models.values()]
labels = list(models.keys())

# Define groups
grouped_models = {
    "FasterMLP": {"indices": [], "color": "red"},
    "EfficientFormer": {"indices": [], "color": "blue"},
    "Independent Models": {"indices": [], "colors": itertools.cycle(['green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'black'])}
}

# Assign indices to groups
for i, label in enumerate(labels):
    if "FasterMLP" in label:
        grouped_models["FasterMLP"]["indices"].append(i)
    elif "EfficientFormer" in label:
        grouped_models["EfficientFormer"]["indices"].append(i)
    else:
        grouped_models["Independent Models"]["indices"].append(i)

# Plot each group
plt.figure(figsize=(10, 6))

# Plot FasterMLP
indices = grouped_models["FasterMLP"]["indices"]
plt.plot(
    [latency[i] for i in indices],
    [top1[i] for i in indices],
    linestyle='-', marker='*', label="FasterMLP (Ours)", color=grouped_models["FasterMLP"]["color"]
)
for i in indices:
    plt.text(latency[i] + 0.5, top1[i], labels[i], fontsize=10, color="red")

# Plot EfficientFormer
indices = grouped_models["EfficientFormer"]["indices"]
plt.plot(
    [latency[i] for i in indices],
    [top1[i] for i in indices],
    linestyle='-', marker='o', label="EfficientFormer", color=grouped_models["EfficientFormer"]["color"]
)
for i in indices:
    plt.text(latency[i] + 0.5, top1[i], labels[i], fontsize=10, color="blue")

# Plot independent models
for i in grouped_models["Independent Models"]["indices"]:
    color = next(grouped_models["Independent Models"]["colors"])
    plt.scatter(latency[i], top1[i], color=color, label=labels[i], s=50)
    plt.text(latency[i] + 0.5, top1[i], labels[i], fontsize=8, color=color)

# Add labels and title
plt.xlabel("Latency (ms)")
plt.ylabel("Top-1 Accuracy (%)")
plt.title("Latency vs. Top-1 Accuracy on ImageNet-1K")
plt.legend(loc="best")
plt.grid(True)
plt.show()
