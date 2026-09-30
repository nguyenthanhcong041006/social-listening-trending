import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, Any
from loguru import logger

def evaluate_trend_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, Any]:
    """
    Evaluates trend prediction model performance on the Test set:
    - Accuracy: Overall classification correctness
    - Precision: Proportion of predicted trending topics that actually trended
    - Recall: Coverage of all actual trending events detected
    - F1-Score: Harmonic mean of Precision and Recall
    - ROC-AUC: Area under the Receiver Operating Characteristic curve
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = 0.5
            
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc) if auc is not None else None,
        "confusion_matrix": cm
    }
    
    logger.info("=== TREND PREDICTION MODEL EVALUATION RESULTS ===")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    if auc is not None:
        logger.info(f"ROC-AUC:   {auc:.4f}")
    logger.info(f"Confusion Matrix:\n{np.array(cm)}")
    
    return metrics
