import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Any, List
from src.baselines import evaluate_predictions

logger = logging.getLogger(__name__)

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
    save_path: str
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Trains a PyTorch neural network model with early stopping based on validation loss.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    logger.info(f"Starting training on device: {device}")
    
    for epoch in range(1, epochs + 1):
        # 1. Train loop
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
            
        train_loss /= total
        train_acc = correct / total
        
        # 2. Validation loop
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item() * len(y_batch)
                preds = torch.argmax(logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        # Compute validation metrics
        num_classes = len(np.unique(val_targets)) if len(val_targets) > 0 else 3
        # Ensure num_classes is at least 3 for multi-class indexing consistency
        num_classes = max(num_classes, 3)
        val_metrics = evaluate_predictions(val_targets, val_preds, num_classes=num_classes)
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        
        logger.info(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Macro F1: {val_f1:.4f}"
        )
        
        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            # Save the model weights
            torch.save(best_model_state, save_path)
            logger.info(f"Saved best model checkpoint with validation loss {val_loss:.4f} to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs. Best Val Loss: {best_val_loss:.4f}")
                break
                
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    elif os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        
    return model, history

def predict_loader(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and probabilities for a PyTorch dataloader.
    """
    model.eval()
    model = model.to(device)
    
    preds_all = []
    probas_all = []
    
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probas = softmax(logits)
            preds = torch.argmax(logits, dim=1)
            
            preds_all.extend(preds.cpu().numpy())
            probas_all.extend(probas.cpu().numpy())
            
    return np.array(preds_all), np.array(probas_all)
