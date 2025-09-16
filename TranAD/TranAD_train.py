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
DATA_PATH = "../dat_dataset_4"
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
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    
    train_losses = {'mse': [], 'mae': [], 'total': []}
    val_losses = {'mse': [], 'mae': [], 'total': []}
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_mse = 0
        train_mae = 0
        train_total = 0
        
        for idx, batch in enumerate(train_loader):
            noisy, clean = batch  # Unpack paired data
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            denoised = model(noisy)
            
            # Calculate losses
            mse_loss = mse_criterion(denoised, clean)
            mae_loss = l1_criterion(denoised, clean)
            total_loss = mse_loss + 0.5 * mae_loss  # Weighted combination
            
            total_loss.backward()
            optimizer.step()
            
            train_mse += mse_loss.item()
            train_mae += mae_loss.item()
            train_total += total_loss.item()
            
        # Average losses
        train_mse /= len(train_loader)
        train_mae /= len(train_loader)
        train_total /= len(train_loader)
        
        train_losses['mse'].append(train_mse)
        train_losses['mae'].append(train_mae)
        train_losses['total'].append(train_total)

        model.eval()
        val_mse = 0
        val_mae = 0
        val_total = 0
        
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                denoised = model(noisy)
                
                mse_loss = mse_criterion(denoised, clean)
                mae_loss = l1_criterion(denoised, clean)
                total_loss = mse_loss + 0.5 * mae_loss
                
                val_mse += mse_loss.item()
                val_mae += mae_loss.item()
                val_total += total_loss.item()
                
        val_mse /= len(val_loader)
        val_mae /= len(val_loader)
        val_total /= len(val_loader)
        
        val_losses['mse'].append(val_mse)
        val_losses['mae'].append(val_mae)
        val_losses['total'].append(val_total)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train MSE: {train_mse:.4f} | Train MAE: {train_mae:.4f} | "
              f"Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f}")

        if val_total < best_loss:
            best_loss = val_total
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "tranad_best.pth"))
            print(f"Saved new best model at epoch {best_epoch} (Total Loss: {best_loss:.4f})")

if __name__ == "__main__":
    train_loader, val_loader = load_train_data(TRAIN_DATA_PATH, VAL_DATA_PATH, BATCH_SIZE)
    model = TranAD()
    train_tranAD(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, DEVICE)
    print("Done")