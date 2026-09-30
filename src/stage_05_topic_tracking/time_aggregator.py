import pandas as pd
from typing import Dict, Any
from loguru import logger

def aggregate_by_time_window(df: pd.DataFrame, time_window: str = "1D") -> pd.DataFrame:
    """
    Groups posts into discrete time intervals (e.g. '1H', '1D', '7D') per topic_id.
    Parses timestamp to standard datetime index.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Exclude outlier noise (topic_id == -1) from tracking
    valid_topics_df = df[df["topic_id"] != -1].copy()
    
    # Floor timestamp to bucket window
    valid_topics_df["time_bucket"] = valid_topics_df["timestamp"].dt.floor(time_window)
    return valid_topics_df
