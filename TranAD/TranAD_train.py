import os
import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

# Ensure project root is on sys.path so imports work when running as a script
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from TranAD.TranAD_model import TranAD
from utils.loss_plots import plot_losses
import pickle
from torch.utils.data import DataLoader as TorchDataLoader
from dat_dataset_4_full.date_loader import DataLoader as EEGDataset

# === DEFAULTS ===
NUM_EPOCHS = 20
MODEL_PATH = "models"
DEFAULT_BATCH_SIZE = 10
DEFAULT_LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            # support both dict and tuple/list batches
            if isinstance(batch, dict):
                noisy, clean = batch['raw'], batch['clean']
            else:
                noisy, clean = batch

            # If shape is (B, F, T) transpose to (B, T, F)
            if noisy.ndim == 3 and noisy.shape[1] < noisy.shape[2]:
                noisy = noisy.transpose(1, 2)
                clean = clean.transpose(1, 2)

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
            for batch in val_loader:
                if isinstance(batch, dict):
                    noisy, clean = batch['raw'], batch['clean']
                else:
                    noisy, clean = batch

                if noisy.ndim == 3 and noisy.shape[1] < noisy.shape[2]:
                    noisy = noisy.transpose(1, 2)
                    clean = clean.transpose(1, 2)

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
            # save_dir is created in main using the timestamped run id
            torch.save(model.state_dict(), os.path.join(model._save_dir, "tranad_best.pth"))
            print(f"Saved new best model at epoch {best_epoch} (Total Loss: {best_loss:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TranAD model")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to dataset folder (local path or mounted input). If not set, uses repo default structure.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--no-cache", action="store_true", help="Disable using cached DataLoaders")

    args = parser.parse_args()

    # Resolve data path: when running inside Azure job the input will be a path provided as --data
    if args.data:
        data_dir = args.data
    else:
        # default to repository data folder
        data_dir = os.path.join(root_dir, "dat_dataset_4")

    print(f"Using data directory: {data_dir}")
    print(f"Batch size: {args.batch_size}, epochs: {args.epochs}, lr: {args.lr}, use_cache: {not args.no_cache}")

    # Create model save directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(MODEL_PATH, f'tranAD_train_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    # Use the dataset loader module from dat_dataset_4_full/date_loader.py
    # Expect the directory `data_dir` to contain 'raw' and 'clean' subfolders as required by the loader.
    try:
        train_dataset = EEGDataset(data_dir, split='training_epochs')
        val_dataset = EEGDataset(data_dir, split='validation_epochs')
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate dataset loader from '{data_dir}': {e}")

    # ensure loader has a transform attribute to avoid AttributeError inside dataset
    setattr(train_dataset, 'transform', None)
    setattr(val_dataset, 'transform', None)

    train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Infer input dims from a single batch
    try:
        sample_batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Training loader is empty; cannot infer input dimensions")

    if isinstance(sample_batch, dict):
        sample_tensor = sample_batch['raw']
    else:
        sample_tensor = sample_batch[0]

    if sample_tensor.ndim == 3:
        b, d1, d2 = sample_tensor.shape
        # If middle dim < last dim assume (B, F, T)
        if d1 < d2:
            num_features = d1
            seq_len = d2
        else:
            seq_len = d1
            num_features = d2
    elif sample_tensor.ndim == 2:
        seq_len, num_features = sample_tensor.shape
    else:
        raise ValueError(f"Unsupported sample tensor shape: {sample_tensor.shape}")

    print(f"Instantiating model with num_features={num_features}, seq_len={seq_len}")
    model = TranAD(num_features=num_features, seq_len=seq_len)
    # attach save_dir to model so training function can access it for saving best model
    setattr(model, "_save_dir", save_dir)

    train_tranAD(model, train_loader, val_loader, args.epochs, args.lr, DEVICE)
    print("Done")