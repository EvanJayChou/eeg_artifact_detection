import torch
import torch.nn as nn


def ensure_btf(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is (B, T, F). Accepts (B, F, T) or (B, T, F)."""
    if t.ndim == 3:
        b, d1, d2 = t.shape
        if d1 < d2:
            t = t.transpose(1, 2)
    return t


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def derivative_charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    dp = torch.diff(pred, dim=1)
    dt = torch.diff(target, dim=1)
    return charbonnier_loss(dp, dt, eps=eps)


def _stft_mag(x: torch.Tensor, n_fft: int, hop_length: int, win_length: int, device: torch.device) -> torch.Tensor:
    B, T, F = x.shape
    xf = x.transpose(1, 2).contiguous().view(B * F, T)  # (B*F, T)
    window = torch.hann_window(win_length, device=device)
    X = torch.stft(xf, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True, return_complex=True)
    mag = torch.abs(X)
    return mag


def multi_resolution_stft_loss(pred: torch.Tensor, target: torch.Tensor, device: torch.device) -> torch.Tensor:
    settings = [
        (64, 32, 64),
        (128, 32, 128),
        (256, 64, 256),
    ]
    total_sc = 0.0
    total_mag = 0.0
    eps = 1e-7
    for n_fft, hop, win in settings:
        S_pred = _stft_mag(pred, n_fft, hop, win, device)
        S_tgt = _stft_mag(target, n_fft, hop, win, device)

        # _stft_mag returns magnitude with leading dim B*F; reshape to per-sample vectors
        B = pred.shape[0]
        diff = S_pred - S_tgt
        diff_flat = diff.contiguous().view(B, -1)
        tgt_flat = S_tgt.contiguous().view(B, -1)

        num = torch.norm(diff_flat, dim=1)  # per-sample norms
        den = torch.norm(tgt_flat, dim=1) + eps
        sc = (num / den).mean()

        log_mag = torch.mean(torch.abs(torch.log(S_pred + eps) - torch.log(S_tgt + eps)))
        total_sc = total_sc + sc
        total_mag = total_mag + log_mag

    n = len(settings)
    return (total_sc + total_mag) / n
