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

from EEG_Conformer_model import ConformerBlock, PositionalEncoding  # reuse building blocks
from utils.loss_plots import plot_losses
from dat_dataset_4.date_loader import DataLoader as EEGDataset


class EEGConformerDenoiser(nn.Module):
    """Sequence-to-sequence Conformer denoiser based on EEG_Conformer building blocks.
    Maps (B, T, F) noisy input to (B, T, F) denoised output.
    """

    def __init__(
        self,
        num_features: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(embed_dim, ff_dim, num_heads, conv_kernel=conv_kernel, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, num_features)

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output_proj(x)
        return x


# === DEFAULTS ===
NUM_EPOCHS = 20
MODEL_PATH = "models"
DEFAULT_BATCH_SIZE = 10
DEFAULT_LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_btf(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is (B, T, F). Accepts (B, F, T) or (B, T, F)."""
    if t.ndim == 3:
        b, d1, d2 = t.shape
        # If middle dim appears to be features (smaller), then transpose to (B, T, F)
        if d1 < d2:  # likely (B, F, T)
            t = t.transpose(1, 2)
        # else already (B, T, F)
    return t


# === TRAINING FUNCTION ===
def train_eeg_conformer(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

    train_losses = {"mse": [], "mae": [], "total": []}
    val_losses = {"mse": [], "mae": [], "total": []}
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_mse = 0.0
        train_mae = 0.0
        train_total = 0.0

        for batch in train_loader:
            noisy, clean = (batch["raw"], batch["clean"]) if isinstance(batch, dict) else batch
            noisy = _ensure_btf(noisy).to(device)
            clean = _ensure_btf(clean).to(device)

            optimizer.zero_grad()
            denoised = model(noisy)

            mse_loss = mse_criterion(denoised, clean)
            mae_loss = l1_criterion(denoised, clean)
            total_loss = mse_loss + 0.5 * mae_loss

            total_loss.backward()
            optimizer.step()

            train_mse += mse_loss.item()
            train_mae += mae_loss.item()
            train_total += total_loss.item()

        n_train = max(1, len(train_loader))
        train_losses["mse"].append(train_mse / n_train)
        train_losses["mae"].append(train_mae / n_train)
        train_losses["total"].append(train_total / n_train)

        # Validation
        model.eval()
        val_mse = 0.0
        val_mae = 0.0
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                noisy, clean = (batch["raw"], batch["clean"]) if isinstance(batch, dict) else batch
                noisy = _ensure_btf(noisy).to(device)
                clean = _ensure_btf(clean).to(device)

                denoised = model(noisy)
                mse_loss = mse_criterion(denoised, clean)
                mae_loss = l1_criterion(denoised, clean)
                total_loss = mse_loss + 0.5 * mae_loss

                val_mse += mse_loss.item()
                val_mae += mae_loss.item()
                val_total += total_loss.item()

        n_val = max(1, len(val_loader))
        val_mse /= n_val
        val_mae /= n_val
        val_total /= n_val

        val_losses["mse"].append(val_mse)
        val_losses["mae"].append(val_mae)
        val_losses["total"].append(val_total)

        print(
            f"Epoch {epoch+1}/{epochs} | Train MSE: {train_losses['mse'][-1]:.4f} | Train MAE: {train_losses['mae'][-1]:.4f} | "
            f"Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f}"
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

    train_eeg_conformer(model, train_loader, val_loader, args.epochs, args.lr, DEVICE)
    print("Done")
