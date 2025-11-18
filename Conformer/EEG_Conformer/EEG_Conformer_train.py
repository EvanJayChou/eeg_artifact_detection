import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader

# Ensure both the Conformer package directory and the repository root are on sys.path
# so imports like `from EEG_Conformer...` and `from utils...` resolve when running the script.
conformer_dir = str(Path(__file__).resolve().parent.parent)  # .../Conformer
repo_root = str(Path(__file__).resolve().parent.parent.parent)  # project root
if conformer_dir not in sys.path:
    # prefer conformer_dir first so `import EEG_Conformer.*` finds the package folder
    sys.path.insert(0, conformer_dir)
if repo_root not in sys.path:
    # keep repo root available for top-level packages like `utils`
    sys.path.insert(1, repo_root)

from EEG_Conformer.EEG_Conformer_denoiser import EEGConformerDenoiser
from EEG_Conformer.EEG_Conformer_utils import ensure_btf, charbonnier_loss, derivative_charbonnier_loss, multi_resolution_stft_loss

from utils.loss_plots import plot_losses
from dat_dataset_4.date_loader import DataLoader as EEGDataset

# === DEFAULTS ===
NUM_EPOCHS = 20
MODEL_PATH = "../models"
DEFAULT_BATCH_SIZE = 10
DEFAULT_LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_W_TIME = 0.5
DEFAULT_W_STFT = 1.0
DEFAULT_W_DERIV = 0.1


_ensure_btf = ensure_btf


def _align_input_for_model(model, noisy, clean):
    """
    Ensure tensors have the channel/features dimension where the model expects it.
    Supports nn.Linear (expects features in last dim) and nn.Conv1d (expects channels in dim=1).
    Returns possibly-permuted (noisy, clean).
    """
    # nothing to do for non-3D tensors
    if noisy.ndim != 3:
        return noisy, clean

    if not hasattr(model, "input_proj"):
        return noisy, clean

    ip = model.input_proj

    # Case: input projection is Linear -> expects last dim == in_features
    if isinstance(ip, nn.Linear):
        expected = ip.in_features
        # if last dim already matches, ok. If second dim matches expected, permute (B, T, F) -> (B, F, T)
        if noisy.shape[-1] != expected and noisy.shape[1] == expected:
            noisy = noisy.permute(0, 2, 1)
            clean = clean.permute(0, 2, 1)
    # Case: input projection is Conv1d -> expects channels at dim=1 == in_channels
    elif isinstance(ip, nn.Conv1d):
        expected = ip.in_channels
        if noisy.shape[1] != expected and noisy.shape[2] == expected:
            noisy = noisy.permute(0, 2, 1)
            clean = clean.permute(0, 2, 1)

    return noisy, clean

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

            # If ensure_btf accidentally returned a 2D tensor (B, T), add feature dim -> (B, T, 1)
            if noisy.ndim == 2:
                noisy = noisy.unsqueeze(-1)
                clean = clean.unsqueeze(-1)

            # Align tensors to model's expected channel/features dimension
            noisy, clean = _align_input_for_model(model, noisy, clean)

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

                # If ensure_btf returned 2D tensors, add feature dim
                if noisy.ndim == 2:
                    noisy = noisy.unsqueeze(-1)
                    clean = clean.unsqueeze(-1)

                # Align tensors to model expectation for validation as well
                noisy, clean = _align_input_for_model(model, noisy, clean)

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
    data_dir = args.data or os.path.join(repo_root, "dat_dataset_4")
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
    # Robust inference: support (B, T, F), (B, F, T), batched 2D (B, T) and single-sample (T, F)
    seq_len = None
    if sample_tensor.ndim == 3:
        b, d1, d2 = sample_tensor.shape
        # decide which axis is features vs time
        if d1 < d2:
            # assume (B, F, T) -> features = d1, seq_len = d2
            num_features = int(d1)
            seq_len = int(d2)
        else:
            # assume (B, T, F)
            seq_len = int(d1)
            num_features = int(d2)
    elif sample_tensor.ndim == 2:
        # Could be batched (B, T) or a single sample (T, F). Use provided batch size to decide.
        bs = args.batch_size
        if sample_tensor.shape[0] == bs:
            # batched 2D: (B, T) -> single-channel sequences
            seq_len = int(sample_tensor.shape[1])
            num_features = 1
        else:
            # single sample (T, F)
            seq_len, num_features = map(int, sample_tensor.shape)
    else:
        raise ValueError(f"Unsupported sample tensor shape: {sample_tensor.shape}")

    print(f"Inferred seq_len={seq_len}, num_features={num_features}")
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
