import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)

class MajorityBaseline:
    def __init__(self):
        self.majority_class = None

    def fit(self, y: np.ndarray):
        classes, counts = np.unique(y, return_counts=True)
        self.majority_class = classes[np.argmax(counts)]
        logger.info(f"MajorityBaseline fitted. Majority class is: {self.majority_class}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.majority_class)

    def predict_proba(self, X: np.ndarray, num_classes: int = 3) -> np.ndarray:
        proba = np.zeros((len(X), num_classes))
        proba[:, self.majority_class] = 1.0
        return proba

class RandomBaseline:
    def __init__(self, random_seed: int = 42):
        self.seed = random_seed
        self.class_probs = None
        self.classes = None

    def fit(self, y: np.ndarray):
        classes, counts = np.unique(y, return_counts=True)
        self.classes = classes
        self.class_probs = counts / len(y)
        logger.info(f"RandomBaseline fitted. Probabilities: {dict(zip(classes, self.class_probs))}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        np.random.seed(self.seed)
        return np.random.choice(self.classes, size=len(X), p=self.class_probs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.class_probs, (len(X), 1))

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None, num_classes: int = 3) -> Dict[str, Any]:
    """
    Evaluates predictions and returns F1, Macro F1, Accuracy, Precision, Recall, ROC-AUC, and Confusion Matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate precision, recall, f1 for macro
    precision_mac, recall_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Calculate precision, recall, f1 per class
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(f1_mac),
        "macro_precision": float(precision_mac),
        "macro_recall": float(recall_mac),
        "f1_per_class": [float(x) for x in f1],
        "precision_per_class": [float(x) for x in precision],
        "recall_per_class": [float(x) for x in recall],
        "confusion_matrix": cm.tolist()
    }
    
    # ROC-AUC calculation
    if y_proba is not None:
        try:
            if num_classes == 2:
                # Binary classification
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                # Multi-class classification (one-vs-rest)
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            metrics["auc"] = float(auc)
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
            metrics["auc"] = 0.5
    else:
        metrics["auc"] = 0.5
        
    return metrics

def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray, 
    X_test: np.ndarray, y_test: np.ndarray,
    feature_cols: list,
    use_cols: list = None
) -> Tuple[LogisticRegression, Dict[str, Any], np.ndarray]:
    """
    Trains a Logistic Regression model on specified features.
    """
    if use_cols is not None:
        indices = [feature_cols.index(col) for col in use_cols if col in feature_cols]
        X_tr = X_train[:, indices]
        X_te = X_test[:, indices]
    else:
        X_tr = X_train
        X_te = X_test

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    # Handle NaNs
    X_tr_scaled = np.nan_to_num(X_tr_scaled, nan=0.0)
    X_te_scaled = np.nan_to_num(X_te_scaled, nan=0.0)

    model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_tr_scaled, y_train)
    
    preds = model.predict(X_te_scaled)
    proba = model.predict_proba(X_te_scaled)
    
    metrics = evaluate_predictions(y_test, preds, proba, num_classes=len(np.unique(y_train)))
    return model, metrics, preds, proba

def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, 
    X_test: np.ndarray, y_test: np.ndarray,
    n_estimators: int = 100
) -> Tuple[RandomForestClassifier, Dict[str, Any], np.ndarray]:
    """
    Trains a Random Forest classifier.
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)
    
    # Handle NaNs
    X_tr_scaled = np.nan_to_num(X_tr_scaled, nan=0.0)
    X_te_scaled = np.nan_to_num(X_te_scaled, nan=0.0)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_tr_scaled, y_train)
    
    preds = model.predict(X_te_scaled)
    proba = model.predict_proba(X_te_scaled)
    
    metrics = evaluate_predictions(y_test, preds, proba, num_classes=len(np.unique(y_train)))
    return model, metrics, preds, proba
