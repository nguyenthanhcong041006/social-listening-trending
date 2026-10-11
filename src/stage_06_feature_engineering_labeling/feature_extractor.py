import pandas as pd
import numpy as np
from src.utils.metrics_utils import compute_acceleration

def extract_social_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sub-stage 6.1: Feature Engineering for Multi-Platform Sports & Early Trend Prediction
    Extracts 4 distinct categories of social listening features:
    
    1. Volumetric & Dynamic Signals:
       - topic_volume, growth_rate, acceleration, volume_lag_1/2/3, growth_rate_lag_1/2/3, rolling_mean_volume_3
    2. Engagement & Social Virality:
       - engagement, comment_to_like_ratio, unique_users, engagement_lag_1/2/3, rolling_mean_engagement_3
    3. Sentiment & Controversy Signals (Sports/Drama/Scandals):
       - avg_sentiment, sentiment_change, sentiment_volatility, positive_ratio, negative_ratio,
       - controversy_score, controversy_lag_1/2/3, sentiment_drop_acceleration, rolling_mean_controversy_3
    4. Cross-Platform Diffusion Signals:
       - platform_count, cross_platform_entropy, platform_spread_speed, fb_volume, tiktok_volume, yt_volume, reddit_volume, news_volume
    """
    df = df.sort_values(by=["topic_id", "time_bucket"]).copy()
    
    # 1. Hashtag Growth & Acceleration
    if "hashtag_activity" in df.columns:
        df["hashtag_growth"] = df.groupby("topic_id")["hashtag_activity"].pct_change().fillna(0.0)
    else:
        df["hashtag_growth"] = 0.0

    if "growth_rate" in df.columns and "acceleration" not in df.columns:
        df["acceleration"] = df.groupby("topic_id")["growth_rate"].diff().fillna(0.0)
    elif "acceleration" in df.columns:
        df["acceleration"] = df["acceleration"].fillna(0.0)
        
    # Sentiment drop acceleration (steep drop in sentiment often heralds sports crisis / scandal)
    if "sentiment_change" in df.columns:
        df["sentiment_drop_acceleration"] = df.groupby("topic_id")["sentiment_change"].diff().fillna(0.0)
    else:
        df["sentiment_drop_acceleration"] = 0.0

    # 2. Temporal Lags (Lags 1, 2, 3 for Early Warning Observation)
    for lag in [1, 2, 3]:
        df[f"volume_lag_{lag}"] = df.groupby("topic_id")["topic_volume"].shift(lag).fillna(0.0)
        df[f"growth_rate_lag_{lag}"] = df.groupby("topic_id")["growth_rate"].shift(lag).fillna(0.0)
        df[f"engagement_lag_{lag}"] = df.groupby("topic_id")["engagement"].shift(lag).fillna(0.0)
        
        if "controversy_score" in df.columns:
            df[f"controversy_lag_{lag}"] = df.groupby("topic_id")["controversy_score"].shift(lag).fillna(0.0)
        if "platform_count" in df.columns:
            df[f"platform_count_lag_{lag}"] = df.groupby("topic_id")["platform_count"].shift(lag).fillna(0.0)
        if "avg_sentiment" in df.columns:
            df[f"sentiment_lag_{lag}"] = df.groupby("topic_id")["avg_sentiment"].shift(lag).fillna(0.0)

    # 3. Rolling Statistics (3-step Moving Averages)
    df["rolling_mean_volume_3"] = (
        df.groupby("topic_id")["topic_volume"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["rolling_mean_engagement_3"] = (
        df.groupby("topic_id")["engagement"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    if "controversy_score" in df.columns:
        df["rolling_mean_controversy_3"] = (
            df.groupby("topic_id")["controversy_score"]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    else:
        df["rolling_mean_controversy_3"] = 0.0
        
    return df
