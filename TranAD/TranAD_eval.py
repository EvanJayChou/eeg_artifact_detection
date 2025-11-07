import os
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
import matplotlib.pyplot as plt

from TranAD.TranAD_model import TranAD
from dat_dataset_4_full.date_loader import DataLoader as EEGDataset


# === Defaults ===
BATCH_SIZE = 10
EVAL_PATH = "evaluations"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_tranAD(model, test_loader, model_path, save_dir, device):
    # Load weights
    state = torch.load(model_path, map_location=device)
    # Support either state_dict or whole model dict
    if isinstance(state, dict) and any(k.startswith('module.') or k in model.state_dict() for k in state.keys()):
        model.load_state_dict(state)
    else:
        # fallback: try loading as state dict directly
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    mse_losses = []
    psnr_values = []
    snr_improvements = []
    correlations = []

    def calculate_snr(signal):
        mean = torch.mean(signal)
        noise = signal - mean
        return 20 * torch.log10(torch.abs(mean) / (torch.std(noise) + 1e-9))

    def calculate_psnr(clean, denoised):
        mse = torch.mean((clean - denoised) ** 2)
        max_val = torch.max(torch.abs(clean)) + 1e-9
        return 20 * torch.log10(max_val / torch.sqrt(mse + 1e-12))

    with torch.no_grad():
        for batch in test_loader:
            # support dict or tuple
            if isinstance(batch, dict):
                noisy, clean = batch['raw'], batch['clean']
            else:
                noisy, clean = batch

            noisy, clean = noisy.to(device), clean.to(device)
            denoised = model(noisy)

            mse = F.mse_loss(denoised, clean).item()
            mse_losses.append(mse)

            psnr = calculate_psnr(clean, denoised)
            psnr_values.append(psnr.item())

            noisy_snr = calculate_snr(noisy)
            denoised_snr = calculate_snr(denoised)
            snr_improvements.append((denoised_snr - noisy_snr).item())

            clean_flat = clean.view(-1).cpu().numpy()
            denoised_flat = denoised.view(-1).cpu().numpy()
            corr = np.corrcoef(clean_flat, denoised_flat)[0, 1]
            correlations.append(corr)

    avg_mse = float(np.mean(mse_losses)) if mse_losses else float('nan')
    avg_psnr = float(np.mean(psnr_values)) if psnr_values else float('nan')
    avg_snr_imp = float(np.mean(snr_improvements)) if snr_improvements else float('nan')
    avg_corr = float(np.mean(correlations)) if correlations else float('nan')
    rmse = float(np.sqrt(avg_mse)) if not np.isnan(avg_mse) else float('nan')

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "report.txt"), "w") as f:
        f.write("=== TranAD EEG Denoising Evaluation Report ===\n")
        f.write(f"MSE: {avg_mse:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SNR Improvement: {avg_snr_imp:.2f} dB\n")
        f.write(f"Signal Correlation: {avg_corr:.4f}\n")
    print(f"Evaluation report saved to {save_dir}")

    # Plot sample reconstructions (take up to 5 batches)
    plt.figure(figsize=(12, 6))
    it = iter(test_loader)
    for i in range(min(5, len(test_loader))):
        try:
            batch = next(it)
        except StopIteration:
            break
        if isinstance(batch, dict):
            noisy, clean = batch['raw'], batch['clean']
        else:
            noisy, clean = batch
        noisy, clean = noisy.to(device), clean.to(device)
        denoised = model(noisy)

        plt.subplot(5, 1, i + 1)
        plt.plot(clean[0].cpu().numpy(), label='Clean', alpha=0.7)
        plt.plot(denoised[0].detach().cpu().numpy(), label='Denoised', alpha=0.7)
        plt.plot(noisy[0].cpu().numpy(), label='Noisy', alpha=0.4)
        if i == 0:
            plt.legend()
        plt.title(f'Sample {i+1}')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reconstructions.png'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate TranAD model")
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root (same layout as training)')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model .pth (state_dict)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--split', type=str, default='testing_epochs', help='Data split to evaluate (testing_epochs or validation_epochs)')
    parser.add_argument('--out', type=str, default=EVAL_PATH, help='Output directory for evaluation report')
    args = parser.parse_args()

    data_dir = args.data
    model_path = args.model
    batch_size = args.batch_size
    split = args.split

    # instantiate dataset
    dataset = EEGDataset(data_dir, split=split)
    setattr(dataset, 'transform', None)
    test_loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Infer model input dims from a batch
    try:
        sample_batch = next(iter(test_loader))
    except StopIteration:
        raise RuntimeError('Test loader is empty; cannot infer model input dimensions')
    if isinstance(sample_batch, dict):
        sample_tensor = sample_batch['raw']
    else:
        sample_tensor = sample_batch[0]
    if sample_tensor.ndim == 3:
        b, d1, d2 = sample_tensor.shape
        if d1 < d2:
            num_features = d1
            seq_len = d2
        else:
            seq_len = d1
            num_features = d2
    elif sample_tensor.ndim == 2:
        seq_len, num_features = sample_tensor.shape
    else:
        raise ValueError(f'Unsupported sample tensor shape: {sample_tensor.shape}')

    model = TranAD(num_features=num_features, seq_len=seq_len)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.out, f'tranAD_eval_{timestamp}')

    evaluate_tranAD(model, test_loader, model_path, save_dir, DEVICE)