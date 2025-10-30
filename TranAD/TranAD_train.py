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
import shutil

# Ensure project root is on sys.path so imports work when running as a script
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from TranAD.TranAD_model import TranAD
from utils.loss_plots import plot_losses
from utils.dataloader import load_train_data
import pickle
from torch.utils.data import DataLoader as TorchDataLoader
from glob import glob

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

    # If the provided data_dir contains cached DataLoader files, copy them into the project's utils cached directory
    project_cache_dir = os.path.join(root_dir, "utils", "cached_loaders")
    os.makedirs(project_cache_dir, exist_ok=True)

    try:
        # If user uploaded a folder that contains a 'cached_loaders' subfolder
        candidate_subdir = os.path.join(data_dir, "cached_loaders")
        if os.path.isdir(candidate_subdir):
            for fname in os.listdir(candidate_subdir):
                if fname.endswith('.pkl'):
                    shutil.copy(os.path.join(candidate_subdir, fname), project_cache_dir)
        else:
            # Otherwise, copy any .pkl files found directly at the mounted data_dir
            for fname in os.listdir(data_dir):
                if fname.endswith('.pkl'):
                    shutil.copy(os.path.join(data_dir, fname), project_cache_dir)
    except Exception as e:
        print(f"No cache files copied: {e}")

    # --- New: if the data_dir already contains pickled dataloaders, load them directly.
    # Assumption: two files are provided named with prefixes 'dataset_loader_clean' and 'dataset_loader_raw'
    # 'dataset_loader_clean*' -> train loader (cleaned/processed), 'dataset_loader_raw*' -> validation loader
    provided_train_loader = None
    provided_val_loader = None
    try:
        # search for files matching possible names
        clean_candidates = glob(os.path.join(data_dir, 'dataset_loader_clean*.pkl'))
        raw_candidates = glob(os.path.join(data_dir, 'dataset_loader_raw*.pkl'))

        def try_load_loader(path):
            try:
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
                # If already a DataLoader, return directly
                if isinstance(obj, TorchDataLoader):
                    return obj
                # If dict with dataset entries, reconstruct DataLoader
                if isinstance(obj, dict):
                    if 'train_dataset' in obj and 'val_dataset' in obj:
                        # this is a combined cache file; caller expects two loaders
                        tr = TorchDataLoader(obj['train_dataset'], batch_size=obj.get('batch_size', DEFAULT_BATCH_SIZE), shuffle=False, num_workers=obj.get('num_workers', 0))
                        vl = TorchDataLoader(obj['val_dataset'], batch_size=obj.get('batch_size', DEFAULT_BATCH_SIZE), shuffle=False, num_workers=obj.get('num_workers', 0))
                        return (tr, vl)
                    if 'train_dataset' in obj:
                        return TorchDataLoader(obj['train_dataset'], batch_size=obj.get('batch_size', DEFAULT_BATCH_SIZE), shuffle=False, num_workers=obj.get('num_workers', 0))
                    if 'val_dataset' in obj:
                        return TorchDataLoader(obj['val_dataset'], batch_size=obj.get('batch_size', DEFAULT_BATCH_SIZE), shuffle=False, num_workers=obj.get('num_workers', 0))
                # If it's a Dataset instance, wrap it
                try:
                    import torch.utils.data as tud
                    if isinstance(obj, tud.Dataset):
                        return TorchDataLoader(obj, batch_size=DEFAULT_BATCH_SIZE, shuffle=False, num_workers=0)
                except Exception:
                    pass
                # unknown object
                return None
            except Exception:
                return None

        # Try loading combined cache first (some users save both in one file)
        combined_candidates = glob(os.path.join(data_dir, 'dataset_loader_*.pkl'))
        for c in combined_candidates:
            loaded = try_load_loader(c)
            if isinstance(loaded, tuple) and len(loaded) == 2:
                provided_train_loader, provided_val_loader = loaded
                break

        if provided_train_loader is None and clean_candidates:
            provided_train_loader = try_load_loader(clean_candidates[0])
        if provided_val_loader is None and raw_candidates:
            provided_val_loader = try_load_loader(raw_candidates[0])
    except Exception as e:
        print(f"Error while attempting to load provided dataloaders: {e}")

    # Use provided pickled dataloaders if present, otherwise fall back to constructing loaders
    if provided_train_loader is not None and provided_val_loader is not None:
        print("Using provided pickled dataloaders from data directory.")
        train_loader = provided_train_loader
        val_loader = provided_val_loader
    else:
        # Use the mounted data path (args.data) if provided, otherwise fall back to repo dataset
        if args.data:
            repo_data_dir = args.data
        else:
            repo_data_dir = os.path.join(root_dir, "dat_dataset_4")

        train_loader, val_loader = load_train_data(repo_data_dir, args.batch_size, use_cache=not args.no_cache)

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