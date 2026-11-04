import csv
import matplotlib.pyplot as plt

with open("training_history.csv") as f:
    data = list(csv.DictReader(f))

epochs = [int(row["epoch"]) for row in data]

# Generate accuracy and loss graphs
for metric in ["accuracy", "loss"]:
    plt.figure()

    for split in ["train", "val"]:
        values = [float(row[f"{split}_{metric}"]) for row in data]

        if metric == "accuracy":
            values = [v * 100 for v in values]

        plt.plot(epochs, values, marker="o", label=split.capitalize())

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)" if metric == "accuracy" else "Loss")
    plt.title(f"Training and Validation {metric.capitalize()}")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"training_{metric}.png")
    plt.close()

print("Training graphs saved.")