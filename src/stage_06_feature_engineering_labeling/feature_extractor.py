import pandas as pd
import numpy as np
from src.utils.metrics_utils import compute_acceleration

def extract_social_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sub-stage 6.1: Feature Engineering
    Extracts Social Listening feature vectors for each topic at each time bucket:
    1. Volume: Discussion post count
    2. Growth Rate: Percentage change in volume
    3. Engagement: Social interaction score
    4. Hashtag Growth: Percentage growth in active hashtags
    5. Sentiment Change: Sentiment score delta
    6. Acceleration: Growth rate acceleration (second derivative of Volume)
    7. Unique Users: Broad reach through distinct author count
    Along with temporal lag features and rolling statistics.
    """
    df = df.sort_values(by=["topic_id", "time_bucket"]).copy()
    
    # 4. Hashtag Growth: Percentage change relative to previous bucket
    df["hashtag_growth"] = df.groupby("topic_id")["hashtag_activity"].pct_change().fillna(0.0)
    
    # 6. Acceleration: Growth acceleration = Growth_Rate_t - Growth_Rate_{t-1}
    df["acceleration"] = df.groupby("topic_id")["growth_rate"].diff().fillna(0.0)
    
    # Lag Features (Temporal lags of Volume, Growth, and Engagement)
    for lag in [1, 2, 3]:
        df[f"volume_lag_{lag}"] = df.groupby("topic_id")["topic_volume"].shift(lag).fillna(0.0)
        df[f"growth_rate_lag_{lag}"] = df.groupby("topic_id")["growth_rate"].shift(lag).fillna(0.0)
        df[f"engagement_lag_{lag}"] = df.groupby("topic_id")["engagement"].shift(lag).fillna(0.0)
        
    # Rolling Statistics (3-period moving averages)
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
    
    return df
