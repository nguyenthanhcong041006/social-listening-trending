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
    
    # Normalize frequency string for pandas 2.2+ (e.g. '6H' -> '6h', '1H' -> '1h', '30T' -> '30min')
    norm_window = time_window
    if isinstance(norm_window, str):
        if norm_window.endswith("H"):
            norm_window = norm_window[:-1] + "h"
        elif norm_window.endswith("T"):
            norm_window = norm_window[:-1] + "min"
        elif norm_window.endswith("S"):
            norm_window = norm_window[:-1] + "s"

    # Floor timestamp to bucket window
    valid_topics_df["time_bucket"] = valid_topics_df["timestamp"].dt.floor(norm_window)
    return valid_topics_df
