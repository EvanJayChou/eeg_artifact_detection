import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from .TranAD_model import TranAD
from ..utils.dataloader import load_test_data
import matplotlib.pyplot as plt

# === GLOBAL VARIABLES ===
BATCH_SIZE = 10
DATA_PATH = "dat_dataset_4"
MODEL_PATH = ""
EVAL_PATH = "evaluations"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory for saving model files
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join(EVAL_PATH, f'tranAD_train_{timestamp}')
os.makedirs(save_dir, exist_ok=True)

# === EVALUATION FUNCTION ===
def evaluate_tranAD(model, test_loader, model_path, save_dir, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    mse_losses = []
    psnr_values = []
    snr_improvements = []
    correlations = []

    def calculate_snr(signal):
        mean = torch.mean(signal)
        noise = signal - mean
        return 20 * torch.log10(torch.abs(mean) / torch.std(noise))
    
    def calculate_psnr(clean, denoised):
        mse = torch.mean((clean - denoised) ** 2)
        max_val = torch.max(clean)
        return 20 * torch.log10(max_val / torch.sqrt(mse))

    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            denoised = model(noisy)

            # Calculate MSE
            mse = F.mse_loss(denoised, clean).item()
            mse_losses.append(mse)

            # Calculate PSNR
            psnr = calculate_psnr(clean, denoised)
            psnr_values.append(psnr.item())

            # Calculate SNR improvement
            noisy_snr = calculate_snr(noisy)
            denoised_snr = calculate_snr(denoised)
            snr_imp = denoised_snr - noisy_snr
            snr_improvements.append(snr_imp.item())

            # Calculate correlation coefficient
            clean_flat = clean.view(-1).cpu().numpy()
            denoised_flat = denoised.view(-1).cpu().numpy()
            corr = np.corrcoef(clean_flat, denoised_flat)[0,1]
            correlations.append(corr)
    
    # Calculate average metrics
    avg_mse = np.mean(mse_losses)
    avg_psnr = np.mean(psnr_values)
    avg_snr_imp = np.mean(snr_improvements)
    avg_corr = np.mean(correlations)
    rmse = np.sqrt(avg_mse)

    # Save evaluation report
    with open(os.path.join(save_dir, "report.txt"), "w") as f:
        f.write("=== TranAD EEG Denoising Evaluation Report ===\n")
        f.write(f"MSE: {avg_mse:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SNR Improvement: {avg_snr_imp:.2f} dB\n")
        f.write(f"Signal Correlation: {avg_corr:.4f}\n")
    print(f"Evaluation report saved to {save_dir}")

    # Plot sample reconstructions
    plt.figure(figsize=(12, 6))
    for i in range(min(5, len(test_loader))):
        noisy, clean = next(iter(test_loader))
        noisy, clean = noisy.to(device), clean.to(device)
        denoised = model(noisy)
        
        plt.subplot(5, 1, i+1)
        plt.plot(clean[0].cpu().numpy(), label='Clean', alpha=0.7)
        plt.plot(denoised[0].detach().cpu().numpy(), label='Denoised', alpha=0.7)
        plt.plot(noisy[0].cpu().numpy(), label='Noisy', alpha=0.4)
        if i == 0:
            plt.legend()
        plt.title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reconstructions.png'))

if __name__ == "__main__":
    test_loader = load_test_data(TEST_DATA_PATH, BATCH_SIZE)
    model = TranAD()
    evaluate_tranAD(model, test_loader, MODEL_PATH, save_dir, DEVICE)