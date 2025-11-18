import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd dimension: last column remains zero for cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # Use actual input sequence length to slice positional encodings (robust)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.dtype)
        return x


class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1, activation=nn.SiLU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvolutionModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "Conv kernel must be odd for same padding"
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=(kernel_size // 2), groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        x = x.transpose(1, 2)  # (batch, dim, seq_len)
        x = self.pointwise_conv1(x)  # (batch, 2*dim, seq_len)
        x = self.glu(x)  # (batch, dim, seq_len)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (batch, seq_len, dim)
        return x


class ConformerBlock(nn.Module):
    """A simplified Conformer block with macaron FFN, MHSA, convolution module, and final FFN."""

    def __init__(self, dim, ff_dim, num_heads, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, ff_dim, dropout=dropout)
        self.ffn2 = FeedForwardModule(dim, ff_dim, dropout=dropout)

        self.norm_ffn1 = nn.LayerNorm(dim)
        self.norm_mha = nn.LayerNorm(dim)
        self.norm_conv = nn.LayerNorm(dim)
        self.norm_ffn2 = nn.LayerNorm(dim)

        # Multi-head self-attention
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # Convolution module
        self.conv_module = ConvolutionModule(dim, kernel_size=conv_kernel, dropout=dropout)

    def forward(self, x, attn_mask=None):
        # x: (batch, seq_len, dim)
        # Macaron FFN (half-step)
        residual = x
        x = self.norm_ffn1(x)
        x = self.ffn1(x)
        x = self.dropout(x)
        x = residual + 0.5 * x

        # Multi-head self-attention
        residual = x
        x = self.norm_mha(x)
        # nn.MultiheadAttention expects (batch, seq_len, embed) when batch_first=True
        attn_out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        x = residual + self.dropout(attn_out)

        # Convolution module
        residual = x
        x = self.norm_conv(x)
        x = self.conv_module(x)
        x = residual + self.dropout(x)

        # Final FFN (half-step)
        residual = x
        x = self.norm_ffn2(x)
        x = self.ffn2(x)
        x = self.dropout(x)
        x = residual + 0.5 * x

        return x


class EEG_Conformer(nn.Module):
    def __init__(self, n_channels=32, n_classes=4, embed_dim=128, num_heads=4, num_layers=2, ff_dim=256, conv_kernel=31, dropout=0.1, max_len=1000):
        super().__init__()

        # === Convolutional Feature Extractor ===
        self.temporal_conv = nn.Conv2d(1, 32, (1, 25), padding=(0, 12))
        self.spatial_conv = nn.Conv2d(32, 64, (n_channels, 1))
        self.pool = nn.AvgPool2d((1, 4))

        # === Token Projection ===
        self.projection = nn.Linear(64, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max_len)

        # === Conformer blocks ===
        self.layers = nn.ModuleList([
            ConformerBlock(embed_dim, ff_dim, num_heads, conv_kernel=conv_kernel, dropout=dropout)
            for _ in range(num_layers)
        ])

        # === Classifier ===
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x, attn_mask=None):
        # x: (batch, channels, samples)
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.pool(x)

        # Flatten spatial dimensions -> tokens
        x = x.squeeze(2).permute(0, 2, 1)  # x: (batch, seq_len, features)
        x = self.projection(x)

        # Add positional encoding
        x = self.pos_enc(x)

        # Conformer stack
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # Pool and classify
        x = x.mean(dim=1)  # global average pooling
        out = self.classifier(x)
        return out