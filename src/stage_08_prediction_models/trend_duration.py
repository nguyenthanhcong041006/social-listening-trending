import numpy as np
import pandas as pd
from typing import Dict, Any


def estimate_trend_duration_and_persistence(
    results_df: pd.DataFrame,
    threshold: float = 0.23
) -> pd.DataFrame:
    """
    Estimates Trend Persistence (24h outlook) and Trend Lifespan (Duration in days).
    
    Parameters:
    - results_df: DataFrame containing model predictions (trending_probability, sarima_forecast_volume, etc.)
    - threshold: Classification decision threshold (tuned on validation set, default 0.23)
    
    Outputs enriched columns:
    - prob_trending_24h: Probability that the topic remains trending after 24 hours.
    - trend_status_24h: Human-readable status ('Sustained Trending (> 24h)', 'Cooling Down (Trend ending in < 24h)', 'Normal (Not Trending)').
    - estimated_duration_days: Projected number of days the topic maintains trending momentum.
    - trend_lifespan: Categorical lifespan classification:
        * '< 24h (Flash Trend)'
        * '1 - 2 Days (Short-term)'
        * '2 - 3 Days (Medium-term)'
        * '>= 3 Days (Sustained)'
    - persistence_score: Percentage score indicating momentum retention into the future.
    """
    df = results_df.copy()
    
    # Extract baseline metrics
    v_current = df["topic_volume"].values.astype(float) if "topic_volume" in df.columns else np.ones(len(df))
    sarima_vol = df["sarima_forecast_volume"].values.astype(float) if "sarima_forecast_volume" in df.columns else v_current * 3.0
    
    # 24h horizon expected volume (Day 1 of SARIMA forecast window)
    v_day1 = sarima_vol / 3.0
    growth_day1 = np.where(v_current > 0, (v_day1 - v_current) / np.maximum(v_current, 1.0), 0.0)
    
    acceleration = df["acceleration"].fillna(0.0).values.astype(float) if "acceleration" in df.columns else np.zeros(len(df))
    acc_clipped = np.clip(acceleration, -2.0, 2.0)
    prob_trend = df["trending_probability"].values.astype(float) if "trending_probability" in df.columns else np.zeros(len(df))
    
    # 1. Calculate 24h Horizon Probability (prob_trending_24h)
    # Combines overall model probability with 24h projected momentum and acceleration
    momentum_factor = 1.0 / (1.0 + np.exp(-1.5 * (growth_day1 + 0.3 * acc_clipped)))
    prob_24h = np.clip(0.60 * prob_trend + 0.40 * momentum_factor * prob_trend * 1.5, 0.01, 0.99)
    
    # 2. Determine 24h Trend Status
    # Evaluates whether the topic will still be trending after 24h
    status_24h = []
    for p24, p_all in zip(prob_24h, prob_trend):
        if p_all >= threshold:
            if p24 >= threshold:
                status_24h.append("Sustained Trending (> 24h)")
            else:
                status_24h.append("Cooling Down (Trend ending in < 24h)")
        else:
            if p24 >= threshold:
                status_24h.append("Emerging Potential (> 24h)")
            else:
                status_24h.append("Normal (Not Trending)")
    
    # 3. Calculate Estimated Duration (in Days)
    # Topics with high probability, positive acceleration and sustained SARIMA forecast receive higher lifespan
    duration_days = np.where(
        prob_trend >= threshold,
        np.clip(
            1.0 + 3.0 * ((prob_trend - threshold) / max(0.42 - threshold, 0.05)) + 0.4 * np.maximum(growth_day1, 0.0),
            1.0,
            5.0
        ),
        np.clip(0.2 + 0.8 * (prob_trend / max(threshold, 0.05)), 0.2, 1.2)
    )
    
    # 4. Classify Trend Lifespan Category
    lifespans = []
    for d, is_tr in zip(duration_days, df["predicted_trending"] if "predicted_trending" in df.columns else [0] * len(df)):
        if is_tr == 0:
            lifespans.append("Not Trending")
        elif d < 1.0:
            lifespans.append("< 24h (Flash Trend)")
        elif d < 2.0:
            lifespans.append("1 - 2 Days (Short-term)")
        elif d < 3.0:
            lifespans.append("2 - 3 Days (Medium-term)")
        else:
            lifespans.append(">= 3 Days (Sustained)")
            
    # 5. Persistence Score (%)
    persistence = np.clip((prob_24h / np.maximum(prob_trend, 0.01)) * 100.0, 10.0, 100.0)
    
    df["prob_trending_24h"] = prob_24h.round(4)
    df["trend_status_24h"] = status_24h
    df["estimated_duration_days"] = duration_days.round(1)
    df["trend_lifespan"] = lifespans
    df["persistence_score"] = persistence.round(1)
    
    return df
