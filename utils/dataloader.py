"""
    WARNING: This dataloader.py script is deprecated and unused.
"""

import torch
from torch.utils.data import DataLoader as TorchDataLoader
import os
import pickle
import sys
from pathlib import Path

# Add the project root to the Python path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from dat_dataset_4.date_loader import DataLoader as EEGDataset

# Set up cache directory relative to this script
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cached_loaders")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dat_dataset_4")
BATCH_SIZE = 10

# Note: Cached DataLoaders are machine-specific and cannot be shared across different computers.
# They require the original dataset files to be present in the same directory structure.
# If moving to a different machine, you'll need to:
# 1. Transfer all the raw EEG data files
# 2. Update DATA_PATH to match the new machine's directory structure
# 3. Create new DataLoaders (the cache will be automatically regenerated)

def save_dataloaders(train_loader, val_loader, cache_path):
    """Save DataLoader objects to disk"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, cache_path)
    
    # Save the dataset and parameters
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'train_dataset': train_loader.dataset,
            'val_dataset': val_loader.dataset,
            'batch_size': train_loader.batch_size,
            'num_workers': train_loader.num_workers
        }, f)
    print(f"Saved DataLoaders to {cache_file}")

def load_cached_loaders(cache_path):
    """Load DataLoader objects from disk"""
    cache_file = os.path.join(CACHE_DIR, cache_path)
    
    if not os.path.exists(cache_file):
        print(f"No cache file found at {cache_file}")
        return None, None
    
    try:
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"Error reading cache file {cache_file}: {str(e)}")
        # Remove corrupted cache file
        os.remove(cache_file)
        return None, None
    
    train_loader = TorchDataLoader(
        cache['train_dataset'],
        batch_size=cache['batch_size'],
        shuffle=False,
        num_workers=cache['num_workers']
    )
    
    val_loader = TorchDataLoader(
        cache['val_dataset'],
        batch_size=cache['batch_size'],
        shuffle=False,  # Don't shuffle validation data
        num_workers=cache['num_workers']
    )
    
    return train_loader, val_loader

def load_train_data(data_dir, batch_size, use_cache=True):
    """Load training data with caching support"""
    cache_path = f"train_val_loaders_{batch_size}.pkl"
    
    if use_cache:
        cached_loaders = load_cached_loaders(cache_path)
        if cached_loaders[0] is not None:
            print("Using cached DataLoaders")
            return cached_loaders

    # Create datasets
    train_dataset = EEGDataset(data_dir, split='training_epochs')
    val_dataset = EEGDataset(data_dir, split='validation_epochs')
    
    # Create data loaders
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Cache the loaders
    if use_cache:
        save_dataloaders(train_loader, val_loader, cache_path)
    
    return train_loader, val_loader

def save_test_loader(test_loader, cache_path):
    """Save test DataLoader object to disk"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, cache_path)
    
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'test_dataset': test_loader.dataset,
            'batch_size': test_loader.batch_size,
            'num_workers': test_loader.num_workers
        }, f)
    print(f"Saved test DataLoader to {cache_file}")

def load_cached_test_loader(cache_path):
    """Load test DataLoader object from disk"""
    cache_file = os.path.join(CACHE_DIR, cache_path)
    
    if not os.path.exists(cache_file):
        print(f"No cache file found at {cache_file}")
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"Error reading cache file {cache_file}: {str(e)}")
        # Remove corrupted cache file
        os.remove(cache_file)
        return None
    
    test_loader = TorchDataLoader(
        cache['test_dataset'],
        batch_size=cache['batch_size'],
        shuffle=False,  # Never shuffle test data
        num_workers=cache['num_workers']
    )
    
    return test_loader

def load_test_data(data_dir, batch_size, use_cache=True):
    """Load test data with caching support"""
    cache_path = f"test_loader_{batch_size}.pkl"
    
    if use_cache:
        cached_loader = load_cached_test_loader(cache_path)
        if cached_loader is not None:
            print("Using cached test DataLoader")
            return cached_loader

    test_dataset = EEGDataset(data_dir, split='testing_epochs')
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Cache the loader
    if use_cache:
        save_test_loader(test_loader, cache_path)
    
    return test_loader

if __name__ == "__main__":
    train_loader, val_loader = load_train_data(DATA_PATH, BATCH_SIZE, use_cache=True)