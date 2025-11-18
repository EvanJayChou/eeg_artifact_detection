import torch.nn as nn
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from .EEG_Conformer_model import ConformerBlock, PositionalEncoding


class EEGConformerDenoiser(nn.Module):
    """Sequence-to-sequence Conformer denoiser.
    Input: (B, T, F) noisy -> Output: (B, T, F) denoised
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
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output_proj(x)
        return x
