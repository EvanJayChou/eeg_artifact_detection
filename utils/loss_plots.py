import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, file_path):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10,5))

    plt.plot(epochs, train_losses, label="Train Loss", color="blue")

    plt.plot(epochs, val_losses, label="Val Loss", color="red")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(file_path)