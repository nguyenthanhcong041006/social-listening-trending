import json
import os
import pandas as pd
from loguru import logger
from .feature_extractor import extract_social_features
from .labeler import generate_trending_labels
from src.utils.file_io import save_dataframe, load_dataframe, ensure_dir

# Canonical Feature Groups definition for Research Benchmarks & Ablation Studies
FEATURES_WITHOUT_SENTIMENT = [
    "topic_volume", "growth_rate", "acceleration", "volume_lag_1", "volume_lag_2", "volume_lag_3",
    "growth_rate_lag_1", "growth_rate_lag_2", "growth_rate_lag_3", "rolling_mean_volume_3",
    "engagement", "log_follower_engagement", "views", "engagement_lag_1", "engagement_lag_2", "rolling_mean_engagement_3",
    "unique_users", "comment_to_like_ratio", "hashtag_growth",
    "platform_count", "cross_platform_entropy", "platform_spread_speed", "platform_count_lag_1",
    "fb_volume", "tiktok_volume", "yt_volume", "reddit_volume", "news_volume"
]

FEATURES_SENTIMENT_ONLY = [
    "avg_sentiment", "sentiment_change", "sentiment_volatility", "positive_ratio", "negative_ratio",
    "neutral_ratio", "controversy_score", "controversy_lag_1", "controversy_lag_2",
    "sentiment_drop_acceleration", "rolling_mean_controversy_3", "sentiment_lag_1",
    "uppercase_intensity"
]

FEATURES_WITH_SENTIMENT = FEATURES_WITHOUT_SENTIMENT + FEATURES_SENTIMENT_ONLY

FEATURES_SINGLE_PLATFORM_FB = [
    "fb_volume", "topic_volume", "growth_rate", "acceleration", "volume_lag_1", "volume_lag_2",
    "engagement", "unique_users", "rolling_mean_volume_3"
]

FEATURE_SETS = {
    "with_sentiment": FEATURES_WITH_SENTIMENT,
    "without_sentiment": FEATURES_WITHOUT_SENTIMENT,
    "single_platform_fb": FEATURES_SINGLE_PLATFORM_FB
}

class FeatureAndLabelBuilder:
    """
    Stage 6: Feature Engineering & Label Data
    Builds the tabular dataset containing engineered multi-platform sports features,
    sentiment/controversy signals, and forward ground-truth Trending labels.
    Exports feature dictionaries for Stage 8 Benchmarks (RQ1, RQ2, RQ3, Sentiment Ablation).
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        labeling_cfg = self.config.get("labeling", {})
        self.future_window_steps = labeling_cfg.get("lead_time_steps", labeling_cfg.get("future_time_window_steps", 3))
        self.future_growth_threshold = labeling_cfg.get("future_growth_threshold", 0.5)
        self.future_volume_threshold = labeling_cfg.get("future_volume_threshold", 5.0)
        self.step_hours = float(labeling_cfg.get("step_hours", 6.0))
        self.output_path = self.config.get("output_path", "data/06_features_labels/features_and_labels.parquet")
        self.feature_sets_meta_path = "data/06_features_labels/feature_sets_metadata.json"

    def run(self, tracked_metrics_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting Stage 6: Feature Engineering & Early Trend Labeling...")
        
        # Branch 1: Multi-Platform Sports Feature Engineering
        featured_df = extract_social_features(tracked_metrics_df)
        
        # Branch 2: Ground-Truth Early Prediction Labeling (Two-Stage Cascade Formulation)
        labeled_df = generate_trending_labels(
            featured_df,
            future_window_steps=self.future_window_steps,
            future_growth_threshold=self.future_growth_threshold,
            future_volume_threshold=self.future_volume_threshold,
            step_hours=self.step_hours
        )
        
        trending_count = int((labeled_df["is_trending"] == 1).sum())
        total_count = len(labeled_df)
        trending_pct = trending_count / max(total_count, 1)
        mean_hours = labeled_df.loc[labeled_df["is_trending"] == 1, "time_to_trend_hours"].mean()
        logger.info(f"Dataset creation complete. Total samples: {total_count}. Trending: {trending_count} ({trending_pct:.1%})")
        logger.info(f"Stage 2 Time-to-Trend Onset Baseline: Average countdown = {mean_hours:.1f}h across positive instances.")
        
        # Save feature sets metadata for Stage 8 Ablations
        ensure_dir(self.feature_sets_meta_path)
        with open(self.feature_sets_meta_path, "w", encoding="utf-8") as f:
            json.dump(FEATURE_SETS, f, indent=2)
            
        save_dataframe(labeled_df, self.output_path)
        logger.info(f"Saved featured & labeled dataset to: {self.output_path}")
        logger.info(f"Exported Feature Sets Metadata (RQ2 & Sentiment Ablation) to: {self.feature_sets_meta_path}")
        return labeled_df

    def run_from_file(self, input_path: str) -> pd.DataFrame:
        df = load_dataframe(input_path)
        return self.run(df)
