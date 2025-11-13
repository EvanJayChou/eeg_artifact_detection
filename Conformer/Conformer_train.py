"""
    WARNING: This model is now deprecated.
    The new model is in the EEG_Conformer directory, in which we are using the EEG Conformer variant instead.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .Conformer_model import ConformerEEG
import matplotlib.pyplot as plt
from datetime import datetime
from ..utils.loss_plots import plot_losses
from ..utils.dataloader import load_train_data

# === GLOBAL VARIABLES ===
NUM_EPOCHS = 20
TRAIN_DATA_PATH = ""
VAL_DATA_PATH = ""
MODEL_PATH = "models"
BATCH_SIZE = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory for saving model files
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join(MODEL_PATH, f'tranAD_train_{timestamp}')
os.makedirs(save_dir, exist_ok=True)

# === TRAINING FUNCTION ===
# def train_conformer(self, train_loader, val_loader, num_epochs, lr, device):