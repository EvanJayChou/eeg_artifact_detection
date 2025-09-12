import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .TranAD_model import TranAD
import matplotlib.pyplot as plt
from datetime import datetime
from ..utils.loss_plots import plot_losses
from ..utils.dataloader import load_train_data

# === GLOBAL VARIABLES ===
NUM_EPOCHS = 20
TRAIN_DATA_PATH = ""
VAL_DATA_PATH = ""
MODEL_PATH = "models"
BATCH_SIZE = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory for saving model files
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join(MODEL_PATH, f'tranAD_train_{timestamp}')
os.makedirs(save_dir, exist_ok=True)

# === TRAINING FUNCTION ===
def train_tranAD(model, train_loader, val_loader, num_epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for idx, batch in enumerate(train_loader):
            x = batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "tranad_best.pth")
            print(f"Saved new best model at epoch {best_epoch} (Loss: {best_loss:.4f})")

if __name__ == "__main__":
    train_loader, val_loader = load_train_data(TRAIN_DATA_PATH, VAL_DATA_PATH, BATCH_SIZE)
    model = TranAD()
    train_tranAD(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, DEVICE)
    print("Done")