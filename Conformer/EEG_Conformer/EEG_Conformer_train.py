import os
import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

# Ensure project root is on sys.path so imports work when running as a script
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from EEG_Conformer_model import EEG_Conformer
from utils.loss_plots import plot_losses
import pickle
from torch.utils.data import DataLoader as TorchDataLoader
from dat_dataset_4.date_loader import DataLoader as EEGDataset

# === DEFAULTS ===
NUM_EPOCHS = 20
MODEL_PATH = "../models"
DEFAULT_BATCH_SIZE = 10
DEFAULT_LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRAINING FUNCTION ===
def train_eeg_conformer(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)