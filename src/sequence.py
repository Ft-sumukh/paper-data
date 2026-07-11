import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class LOBSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int):
        """
        PyTorch Dataset for Limit Order Book rolling sequences.
        Inputs:
            X: Normalized feature matrix of shape (N, F)
            y: Label array of shape (N,)
            sequence_length: T, the lookback window size
        """
        assert len(X) == len(y), "X and y must have the same length"
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.sequence_length = sequence_length
        self.num_samples = len(X) - sequence_length + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract features from idx to idx + T
        x_seq = self.X[idx : idx + self.sequence_length]
        # Label is the target for the last event in the sequence window
        y_val = self.y[idx + self.sequence_length - 1]
        return x_seq, y_val

def create_dataloaders(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    sequence_length: int,
    batch_size: int,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Fits StandardScaler on the training features, normalizes all features,
    and returns PyTorch DataLoaders for train, validation, and test splits.
    """
    # 1. Fit scaler on training features ONLY
    scaler = StandardScaler()
    
    # Check for NaN or Inf and fill them
    train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
    val_features = np.nan_to_num(val_features, nan=0.0, posinf=0.0, neginf=0.0)
    test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train_scaled = scaler.fit_transform(train_features)
    X_val_scaled = scaler.transform(val_features)
    X_test_scaled = scaler.transform(test_features)
    
    # 2. Create Dataset objects
    train_dataset = LOBSequenceDataset(X_train_scaled, train_labels, sequence_length)
    val_dataset = LOBSequenceDataset(X_val_scaled, val_labels, sequence_length)
    test_dataset = LOBSequenceDataset(X_test_scaled, test_labels, sequence_length)
    
    # 3. Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(
        f"DataLoaders created. Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )
    
    return train_loader, val_loader, test_loader, scaler
