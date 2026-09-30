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
