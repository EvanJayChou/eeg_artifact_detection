import torch
import torch.nn as nn
import torch.optim as optim

# === TranAD ARCHITECTURE ===
class TranAD(nn.Module):
    def __init__(self, num_features, seq_len, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, num_features)

    def forward(self, x):
        x_proj = self.input_proj(x)
        x_proj = x_proj.permute(1,0,2)

        enc_out = self.encoder(x_proj)
        dec_out = self.decoder(x_proj, enc_out)

        dec_out = dec_out.permute(1,0,2)
        out = self.output_layer(dec_out)
        return out