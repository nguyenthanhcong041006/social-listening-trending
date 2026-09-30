import numpy as np
import pandas as pd
from typing import Dict

def compute_growth_rate(current_volume: float, previous_volume: float) -> float:
    """
    Computes the growth rate:
        Growth Rate = (V_t - V_{t-1}) / max(V_{t-1}, 1)
    """
    if previous_volume <= 0:
        return float(current_volume) if current_volume > 0 else 0.0
    return float((current_volume - previous_volume) / previous_volume)

def compute_acceleration(current_growth: float, previous_growth: float) -> float:
    """
    Computes the growth acceleration:
        Acceleration = Growth_t - Growth_{t-1}
    """
    return float(current_growth - previous_growth)

def compute_engagement_score(
    likes: float,
    shares: float,
    comments: float,
    weights: Dict[str, float] = None
) -> float:
    """
    Computes the weighted engagement score:
        Engagement = w_likes * Likes + w_shares * Shares + w_comments * Comments
    """
    if weights is None:
        weights = {"likes": 1.0, "shares": 2.0, "comments": 1.5}
    
    return float(
        likes * weights.get("likes", 1.0) +
        shares * weights.get("shares", 2.0) +
        comments * weights.get("comments", 1.5)
    )

def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE):
    Bounded between 0% and 200%. Handles zero/low volume time-series gracefully without exploding.
    Formula (PDF Section 11.1 & 12):
        sMAPE = (100 / n) * Σ (|y - y_hat| / ((|y| + |y_hat|) / 2))
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    mask = denominator > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(diff[mask] / denominator[mask]) * 100.0)

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))

def compute_recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k_pct: float = 0.20) -> float:
    """
    Recall@K (PDF Section 12):
    Measures the proportion of actual trending topics captured within the top K highest-probability predictions.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    total_positives = np.sum(y_true == 1)
    if total_positives == 0:
        return 1.0
    k = max(1, int(len(y_prob) * k_pct))
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    captured_positives = np.sum(y_true[top_k_indices] == 1)
    return float(captured_positives / total_positives)
