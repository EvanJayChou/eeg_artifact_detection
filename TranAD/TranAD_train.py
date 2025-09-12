import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# === GLOBAL VARIABLES ===
NUM_EPOCHS = 20
TRAIN_DATA_PATH = ""
VAL_DATA_PATH = ""
MODEL_PATH = "models"
BATCH_SIZE = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join(MODEL_PATH, f'tranAD_train_{timestamp}')
os.makedirs(save_dir, exist_ok=True)

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

# === DATA LOADING FUNCTION ===
def load_data(train_path, val_path, batch_size):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return 

# === TRAINING FUNCTION ===
def train_tranad(model, train_loader, val_loader, num_epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for idx, batch in enumerate(train_loader):
            x = batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "tranad_best.pth")
            print(f"Saved new best model at epoch {best_epoch} (Loss: {best_loss:.4f})")

if __name__ == "__main__":
    train_loader, val_loader = load_data(TRAIN_DATA_PATH, VAL_DATA_PATH, BATCH_SIZE)
    model = TranAD()
