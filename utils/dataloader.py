from torch.utils.data import Dataset, DataLoader

# === DATA LOADING FUNCTIONS ===
def load_train_data(train_path, val_path, batch_size):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

def load_test_data(test_path, batch_size):
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return test_loader