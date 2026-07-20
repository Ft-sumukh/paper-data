import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from src.sequence import LOBSequenceDataset

logger = logging.getLogger(__name__)

def get_logistic_regression_coefs(model, feature_names: List[str]) -> Dict[str, Any]:
    """
    Extracts the feature coefficients for a fitted multi-class Logistic Regression.
    """
    coefs = model.coef_  # Shape: (num_classes, num_features)
    classes = model.classes_
    
    coef_dict = {}
    for i, cls in enumerate(classes):
        cls_name = f"class_{cls}"
        if cls == 0:
            cls_name = "DOWN"
        elif cls == 1:
            cls_name = "STATIONARY"
        elif cls == 2:
            cls_name = "UP"
            
        cls_coefs = dict(zip(feature_names, [float(x) for x in coefs[i]]))
        # Sort by absolute value
        sorted_coefs = dict(sorted(cls_coefs.items(), key=lambda item: abs(item[1]), reverse=True))
        coef_dict[cls_name] = sorted_coefs
        
    return coef_dict

def get_random_forest_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """
    Extracts feature importances from a Random Forest classifier.
    """
    importances = model.feature_importances_
    imp_dict = dict(zip(feature_names, [float(x) for x in importances]))
    return dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True))

def permutation_importance_pytorch(
    model: torch.nn.Module,
    features_raw: np.ndarray,
    labels: np.ndarray,
    scaler: Any,
    sequence_length: int,
    base_f1: float,
    feature_names: List[str],
    device: torch.device,
    batch_size: int = 128
) -> Dict[str, float]:
    """
    Calculates Permutation Feature Importance for a PyTorch sequence classifier.
    Shuffles each feature column in raw space, normalizes, generates sequences,
    obtains predictions, and measures the drop in macro F1-score.
    """
    model.eval()
    importances = {}
    
    # Run evaluation function helper
    def evaluate_model_on_raw(raw_feat: np.ndarray) -> float:
        scaled_feat = scaler.transform(raw_feat)
        scaled_feat = np.nan_to_num(scaled_feat, nan=0.0)
        
        dataset = LOBSequenceDataset(scaled_feat, labels, sequence_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        preds_all = []
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(device)
                logits = model(x_batch)
                preds = torch.argmax(logits, dim=1)
                preds_all.extend(preds.cpu().numpy())
                
        # Handle labels offset due to sequence length
        label_offset = sequence_length - 1
        y_true = labels[label_offset:]
        y_pred = np.array(preds_all)
        
        return f1_score(y_true, y_pred, average='macro', zero_division=0)

    logger.info("Computing Permutation Feature Importance for neural model...")
    for j, feat_name in enumerate(feature_names):
        # Create copy and shuffle the j-th feature column
        shuffled_features = features_raw.copy()
        np.random.shuffle(shuffled_features[:, j])
        
        shuffled_f1 = evaluate_model_on_raw(shuffled_features)
        
        # Importance is the decline in F1 score
        importance = base_f1 - shuffled_f1
        importances[feat_name] = float(importance)
        
    # Sort importances descending
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    logger.info("Finished computing Permutation Feature Importance.")
    return sorted_importances
