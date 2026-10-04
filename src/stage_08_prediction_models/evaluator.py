import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    brier_score_loss,
    balanced_accuracy_score,
    log_loss
)
from typing import Dict, Any, Optional
from loguru import logger
from src.utils.metrics_utils import compute_recall_at_k

def evaluate_trend_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Comprehensive academic and operational evaluation for trend prediction models:
    - Accuracy: Overall classification correctness
    - Balanced Accuracy: Mean of recall obtained on each class (deals with imbalance)
    - Macro-F1: Unweighted mean F1 across classes (robust to class imbalance)
    - Precision: Proportion of predicted trending topics that actually trended
    - Recall: Coverage of all actual trending events detected
    - F1-Score: Binary harmonic mean for the trending class
    - Matthews Correlation Coefficient (MCC): Premier metric for imbalanced binary evaluation in academic papers
    - ROC-AUC: Area under the Receiver Operating Characteristic curve
    - PR-AUC / AP: Area under Precision-Recall curve
    - Brier Score: Calibration mean squared error (lower is better)
    - Log-Loss: Cross-entropy penalty
    - Recall@10% & Recall@20%: Proportions of true trends captured in top-k alert budgets
    """
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    auc = 0.5
    pr_auc = rec
    brier = 0.0
    cross_entropy = 0.0
    recall_10 = 0.0
    recall_20 = 0.0

    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = float(rec)
        try:
            brier = float(brier_score_loss(y_true, y_prob))
        except Exception:
            brier = 0.0
        try:
            cross_entropy = float(log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15)))
        except Exception:
            cross_entropy = 0.0

        recall_10 = float(compute_recall_at_k(y_true, y_prob, k_pct=0.10))
        recall_20 = float(compute_recall_at_k(y_true, y_prob, k_pct=0.20))
            
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    metrics = {
        "model_name": model_name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "mcc": mcc,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": cross_entropy,
        "recall_at_10pct": recall_10,
        "recall_at_20pct": recall_20,
        "confusion_matrix": cm
    }
    
    logger.info(f"=== [{model_name}] EVALUATION RESULTS ===")
    logger.info(f"Macro-F1:    {macro_f1:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")
    logger.info(f"Precision:   {prec:.4f} | Recall: {rec:.4f} | Accuracy: {acc:.4f}")
    logger.info(f"ROC-AUC:     {auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")
    logger.info(f"Recall@10%:  {recall_10:.4f} | Recall@20%: {recall_20:.4f}")
    
    return metrics
