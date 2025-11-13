import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader

# Ensure project root is on sys.path so imports work when running as a script
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from EEG_Conformer_denoiser import EEGConformerDenoiser
from EEG_Conformer_utils import ensure_btf, charbonnier_loss, derivative_charbonnier_loss, multi_resolution_stft_loss
from utils.loss_plots import plot_losses
from dat_dataset_4.date_loader import DataLoader as EEGDataset

# === DEFAULTS ===
NUM_EPOCHS = 20
MODEL_PATH = "models"
DEFAULT_BATCH_SIZE = 10
DEFAULT_LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_W_TIME = 0.5
DEFAULT_W_STFT = 1.0
DEFAULT_W_DERIV = 0.1


_ensure_btf = ensure_btf

# === TRAINING FUNCTION ===
def train_eeg_conformer(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    w_time: float = DEFAULT_W_TIME,
    w_stft: float = DEFAULT_W_STFT,
    w_deriv: float = DEFAULT_W_DERIV,
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = {"time": [], "stft": [], "deriv": [], "total": []}
    val_losses = {"time": [], "stft": [], "deriv": [], "total": []}
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_time = 0.0
        train_stft = 0.0
        train_deriv = 0.0
        train_total = 0.0

        for batch in train_loader:
            noisy, clean = (batch["raw"], batch["clean"]) if isinstance(batch, dict) else batch
            noisy = _ensure_btf(noisy).to(device)
            clean = _ensure_btf(clean).to(device)

            optimizer.zero_grad()
            denoised = model(noisy)

            loss_time = charbonnier_loss(denoised, clean)
            loss_stft = multi_resolution_stft_loss(denoised, clean, device)
            loss_deriv = derivative_charbonnier_loss(denoised, clean)
            total_loss = w_time * loss_time + w_stft * loss_stft + w_deriv * loss_deriv

            total_loss.backward()
            optimizer.step()

            train_time += loss_time.item()
            train_stft += loss_stft.item()
            train_deriv += loss_deriv.item()
            train_total += total_loss.item()

        n_train = max(1, len(train_loader))
        train_losses["time"].append(train_time / n_train)
        train_losses["stft"].append(train_stft / n_train)
        train_losses["deriv"].append(train_deriv / n_train)
        train_losses["total"].append(train_total / n_train)

        # Validation
        model.eval()
        val_time = 0.0
        val_stft = 0.0
        val_deriv = 0.0
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                noisy, clean = (batch["raw"], batch["clean"]) if isinstance(batch, dict) else batch
                noisy = _ensure_btf(noisy).to(device)
                clean = _ensure_btf(clean).to(device)

                denoised = model(noisy)
                loss_time = charbonnier_loss(denoised, clean)
                loss_stft = multi_resolution_stft_loss(denoised, clean, device)
                loss_deriv = derivative_charbonnier_loss(denoised, clean)
                total_loss = w_time * loss_time + w_stft * loss_stft + w_deriv * loss_deriv

                val_time += loss_time.item()
                val_stft += loss_stft.item()
                val_deriv += loss_deriv.item()
                val_total += total_loss.item()

        n_val = max(1, len(val_loader))
        val_time /= n_val
        val_stft /= n_val
        val_deriv /= n_val
        val_total /= n_val

        val_losses["time"].append(val_time)
        val_losses["stft"].append(val_stft)
        val_losses["deriv"].append(val_deriv)
        val_losses["total"].append(val_total)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train (time:{train_losses['time'][-1]:.4f}, stft:{train_losses['stft'][-1]:.4f}, deriv:{train_losses['deriv'][-1]:.4f}) | "
            f"Val (time:{val_time:.4f}, stft:{val_stft:.4f}, deriv:{val_deriv:.4f})"
        )

        # Save best model
        if val_total < best_loss:
            best_loss = val_total
            torch.save(model.state_dict(), os.path.join(model._save_dir, "eeg_conformer_best.pth"))
            print(f"Saved new best model (Val Total: {best_loss:.4f})")

    # Save loss plot
    plot_losses(train_losses["total"], val_losses["total"], os.path.join(model._save_dir, "loss.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG Conformer (denoising)")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset root. If not set, uses repo default.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--w-time", type=float, default=DEFAULT_W_TIME, help="Weight for time-domain Charbonnier loss")
    parser.add_argument("--w-stft", type=float, default=DEFAULT_W_STFT, help="Weight for multi-resolution STFT loss")
    parser.add_argument("--w-deriv", type=float, default=DEFAULT_W_DERIV, help="Weight for derivative Charbonnier loss")
    args = parser.parse_args()

    # Resolve data path
    data_dir = args.data or os.path.join(root_dir, "dat_dataset_4/dat_dataset_4")
    print(f"Using data directory: {data_dir}")
    print(f"Batch size: {args.batch_size}, epochs: {args.epochs}, lr: {args.lr}")

    # Create model save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(MODEL_PATH, f"eeg_conformer_train_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Datasets and loaders
    train_dataset = EEGDataset(data_dir, split="training_epochs")
    val_dataset = EEGDataset(data_dir, split="validation_epochs")
    setattr(train_dataset, "transform", None)
    setattr(val_dataset, "transform", None)
    train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Infer num_features from one batch (we'll operate as (B, T, F))
    try:
        sample_batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Training loader is empty; cannot infer input dimensions")
    sample_tensor = sample_batch["raw"] if isinstance(sample_batch, dict) else sample_batch[0]
    if sample_tensor.ndim == 3:
        b, d1, d2 = sample_tensor.shape
        if d1 < d2:  # (B, F, T) -> we'll transpose to (B, T, F)
            num_features = d1
        else:  # (B, T, F)
            num_features = d2
    elif sample_tensor.ndim == 2:
        # (T, F)
        num_features = sample_tensor.shape[1]
    else:
        raise ValueError(f"Unsupported sample tensor shape: {sample_tensor.shape}")

    print(f"Instantiating EEG Conformer denoiser with num_features={num_features}")
    model = EEGConformerDenoiser(num_features=num_features)
    setattr(model, "_save_dir", save_dir)

    train_eeg_conformer(
        model,
        train_loader,
        val_loader,
        args.epochs,
        args.lr,
        DEVICE,
        w_time=args.w_time,
        w_stft=args.w_stft,
        w_deriv=args.w_deriv,
    )
    print("Done")
