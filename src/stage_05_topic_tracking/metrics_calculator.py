import pandas as pd
import numpy as np
from typing import Dict
from src.utils.metrics_utils import compute_engagement_score, compute_growth_rate

def compute_topic_tracking_metrics(
    grouped_df: pd.DataFrame,
    engagement_weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Topic Tracking Metrics:
    Calculates the 6 time-series tracking dimensions reflecting topic dynamics over time:
    1. Topic Volume: Post count within time bucket
    2. Growth Rate: Volume percentage growth relative to previous time bucket
    3. Engagement: Weighted interaction score (Likes + Shares + Comments)
    4. Sentiment Change: Delta of mean sentiment score between consecutive time buckets
    5. Hashtag Activity: Unique hashtag count active within the window
    6. Unique Users: Distinct post author count
    """
    records = []
    
    for (topic_id, bucket), group in grouped_df.groupby(["topic_id", "time_bucket"]):
        volume = len(group)
        likes = group["likes"].sum() if "likes" in group else 0
        shares = group["shares"].sum() if "shares" in group else 0
        comments = group["comments"].sum() if "comments" in group else 0
        engagement = compute_engagement_score(likes, shares, comments, engagement_weights)
        
        # Average sentiment score
        avg_sentiment = group["sentiment_score"].mean() if "sentiment_score" in group else 0.0
        
        # Hashtag count
        hashtags = []
        for h in group["hashtags"].dropna():
            if isinstance(h, list):
                hashtags.extend(h)
            elif isinstance(h, str):
                hashtags.append(h)
        hashtag_activity = len(set(hashtags))
        
        # Unique users
        unique_users = group["post_id"].nunique() if "post_id" in group else volume
        
        records.append({
            "topic_id": topic_id,
            "time_bucket": bucket,
            "topic_volume": volume,
            "engagement": engagement,
            "avg_sentiment": avg_sentiment,
            "hashtag_activity": hashtag_activity,
            "unique_users": unique_users
        })
        
    metrics_df = pd.DataFrame(records)
    if metrics_df.empty:
        return metrics_df
    
    # Sort chronologically per topic to compute Growth Rate and Sentiment Change deltas
    metrics_df = metrics_df.sort_values(by=["topic_id", "time_bucket"]).reset_index(drop=True)
    
    metrics_df["growth_rate"] = metrics_df.groupby("topic_id")["topic_volume"].pct_change().fillna(0.0)
    metrics_df["sentiment_change"] = metrics_df.groupby("topic_id")["avg_sentiment"].diff().fillna(0.0)
    
    return metrics_df
