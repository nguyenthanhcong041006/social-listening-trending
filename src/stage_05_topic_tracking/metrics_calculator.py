import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.utils.metrics_utils import (
    compute_engagement_score,
    compute_log_follower_engagement,
    compute_growth_rate
)

def compute_topic_tracking_metrics(
    grouped_df: pd.DataFrame,
    engagement_weights: Dict[str, float] = None,
    platform_weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Topic Tracking Metrics with Cross-Platform & Sports Controversy Dimensions:
    Calculates comprehensive time-series tracking dimensions:
    1. Topic Volume: Post count within time bucket
    2. Growth Rate & Acceleration: Volume dynamics and second derivative
    3. Cross-Platform Diffusion: platform_count, cross_platform_entropy, per-platform volumes
    4. Engagement & Follower-Normalized Virality:
       - engagement: Weighted interaction sum
       - log_follower_engagement: Scaled by log10(max(Followers, 100)) and platform weights
       - views: Video impressions across platforms
       - comment_to_like_ratio: Heated debate indicator
    5. Sentiment Dynamics: Mean sentiment, sentiment change, sentiment volatility
    6. Controversy & Intensity Signals: Polarized debate metric and uppercase intensity
    7. Hashtag Activity & Unique Authors: Topic breadth and reach
    """
    if platform_weights is None:
        platform_weights = {
            "facebook": 1.0, "tiktok": 0.8, "youtube": 1.2, "reddit": 1.5, "news": 2.0
        }
    records = []
    
    for (topic_id, bucket), group in grouped_df.groupby(["topic_id", "time_bucket"]):
        volume = len(group)
        likes = group["likes"].sum() if "likes" in group else 0
        shares = group["shares"].sum() if "shares" in group else 0
        comments = group["comments"].sum() if "comments" in group else 0
        views = group["views"].sum() if "views" in group else 0
        
        # Standard weighted engagement & follower-normalized engagement rate
        engagement = compute_engagement_score(likes, shares, comments, engagement_weights)
        comment_to_like_ratio = float(comments / (likes + 1.0))
        
        post_log_engs = []
        for _, row in group.iterrows():
            p_name = str(row.get("platform", "generic")).lower()
            p_weight = platform_weights.get(p_name, 1.0)
            fol = row.get("followers", 0)
            post_log_engs.append(
                compute_log_follower_engagement(
                    likes=row.get("likes", 0),
                    shares=row.get("shares", 0),
                    comments=row.get("comments", 0),
                    followers=fol,
                    weights=engagement_weights,
                    platform_weight=p_weight
                )
            )
        log_follower_engagement = float(np.sum(post_log_engs))
        uppercase_intensity = float(group["uppercase_ratio"].mean()) if "uppercase_ratio" in group else 0.0
        
        # Multi-platform diffusion metrics
        if "platform" in group:
            platform_count = int(group["platform"].nunique())
            p_counts = group["platform"].value_counts(normalize=True).values
            entropy = -float(np.sum(p_counts * np.log2(p_counts + 1e-12)))
            max_entropy = np.log2(5.0)  # normalized across 5 primary platforms (fb, tt, yt, reddit, news)
            cross_platform_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))
            
            fb_vol = int((group["platform"] == "facebook").sum())
            tiktok_vol = int((group["platform"] == "tiktok").sum())
            yt_vol = int((group["platform"] == "youtube").sum())
            reddit_vol = int((group["platform"] == "reddit").sum())
            news_vol = int((group["platform"] == "news").sum())
        else:
            platform_count = 1
            cross_platform_entropy = 0.0
            fb_vol = volume
            tiktok_vol = 0
            yt_vol = 0
            reddit_vol = 0
            news_vol = 0
            
        # Average sentiment score and class ratios
        avg_sentiment = float(group["sentiment_score"].mean()) if "sentiment_score" in group else 0.0
        sentiment_volatility = float(group["sentiment_score"].std(ddof=0)) if ("sentiment_score" in group and len(group) > 1) else 0.0
        
        if "sentiment_label" in group:
            pos_ratio = float((group["sentiment_label"] == "positive").mean())
            neg_ratio = float((group["sentiment_label"] == "negative").mean())
            neu_ratio = float((group["sentiment_label"] == "neutral").mean())
        elif "sentiment_score" in group:
            scores = group["sentiment_score"]
            pos_ratio = float((scores > 0.05).mean())
            neg_ratio = float((scores < -0.05).mean())
            neu_ratio = float(((scores >= -0.05) & (scores <= 0.05)).mean())
        else:
            pos_ratio, neg_ratio, neu_ratio = 0.0, 0.0, 1.0

        # Controversy index (polarization between fans in sports debates / scandal spikes)
        if "controversy_score" in group:
            controversy_score = float(group["controversy_score"].mean())
        else:
            polarization = 2.0 * min(pos_ratio, neg_ratio) / (pos_ratio + neg_ratio + 1e-6)
            controversy_score = float(polarization * (1.0 - neu_ratio))
        
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
            "log_follower_engagement": log_follower_engagement,
            "views": views,
            "uppercase_intensity": uppercase_intensity,
            "comment_to_like_ratio": comment_to_like_ratio,
            "platform_count": platform_count,
            "cross_platform_entropy": cross_platform_entropy,
            "fb_volume": fb_vol,
            "tiktok_volume": tiktok_vol,
            "yt_volume": yt_vol,
            "reddit_volume": reddit_vol,
            "news_volume": news_vol,
            "avg_sentiment": avg_sentiment,
            "sentiment_volatility": sentiment_volatility,
            "positive_ratio": pos_ratio,
            "negative_ratio": neg_ratio,
            "neutral_ratio": neu_ratio,
            "controversy_score": controversy_score,
            "hashtag_activity": hashtag_activity,
            "unique_users": unique_users
        })
        
    metrics_df = pd.DataFrame(records)
    if metrics_df.empty:
        return metrics_df
    
    # Sort chronologically per topic to compute dynamic deltas
    metrics_df = metrics_df.sort_values(by=["topic_id", "time_bucket"]).reset_index(drop=True)
    
    # Growth rate and acceleration (first and second derivatives)
    metrics_df["growth_rate"] = metrics_df.groupby("topic_id")["topic_volume"].pct_change().fillna(0.0)
    metrics_df["acceleration"] = metrics_df.groupby("topic_id")["growth_rate"].diff().fillna(0.0)
    
    # Sentiment & controversy deltas
    metrics_df["sentiment_change"] = metrics_df.groupby("topic_id")["avg_sentiment"].diff().fillna(0.0)
    metrics_df["controversy_change"] = metrics_df.groupby("topic_id")["controversy_score"].diff().fillna(0.0)
    metrics_df["platform_spread_speed"] = metrics_df.groupby("topic_id")["platform_count"].diff().fillna(0.0)
    
    return metrics_df
