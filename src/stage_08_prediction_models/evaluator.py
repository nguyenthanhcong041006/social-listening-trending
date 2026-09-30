import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from typing import Dict, Any
from loguru import logger
from src.utils.metrics_utils import compute_recall_at_k

def evaluate_trend_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, Any]:
    """
    Evaluates trend prediction model performance on the Test set (PDF Section 12):
    - Accuracy: Overall classification correctness
    - Macro-F1: Unweighted mean F1 across classes (robust to class imbalance)
    - Precision: Proportion of predicted trending topics that actually trended
    - Recall: Coverage of all actual trending events detected
    - F1-Score: Binary harmonic mean for the trending class
    - ROC-AUC: Area under the Receiver Operating Characteristic curve
    - PR-AUC: Area under Precision-Recall Curve (crucial for imbalanced trending detection)
    - Recall@K: Proportion of actual trends detected in top K% predictions
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    auc = None
    pr_auc = None
    recall_k = None
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = float(rec)
        recall_k = float(compute_recall_at_k(y_true, y_prob, k_pct=0.20))
            
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "recall_at_20pct": recall_k,
        "confusion_matrix": cm
    }
    
    logger.info("=== TREND PREDICTION MODEL EVALUATION RESULTS (PDF Section 12) ===")
    logger.info(f"Accuracy:    {acc:.4f}")
    logger.info(f"Macro-F1:    {macro_f1:.4f}")
    logger.info(f"Precision:   {prec:.4f}")
    logger.info(f"Recall:      {rec:.4f}")
    logger.info(f"F1-Score:    {f1:.4f}")
    if auc is not None:
        logger.info(f"ROC-AUC:     {auc:.4f}")
    if pr_auc is not None:
        logger.info(f"PR-AUC:      {pr_auc:.4f}")
    if recall_k is not None:
        logger.info(f"Recall@20%:  {recall_k:.4f}")
    logger.info(f"Confusion Matrix:\n{np.array(cm)}")
    
    return metrics
