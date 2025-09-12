import torch
import torch.nn as nn
import torch.optim as optim

class ConformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4, conv_kernel=31):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel//2, groups=d_model),
            nn.GLU(dim=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1)
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + 0.5*self.ffn1(x)
        attn_out, _ = self.self_attn(x,x,x)
        x = x + attn_out
        conv_in = x.transpose(1,2)
        conv_out = self.conv(conv_in).transpose(1,2)
        x = x + conv_out
        x = x + 0.5*self.ffn2(x)
        return self.norm(x)

class ConformerEEG(nn.Module):
    def __init__(self, num_features, num_classes, d_model=64, n_blocks=2):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.blocks = nn.ModuleList([ConformerBlock(d_model) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1,2)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)