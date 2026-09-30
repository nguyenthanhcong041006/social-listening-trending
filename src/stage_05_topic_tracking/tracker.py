import pandas as pd
from loguru import logger
from .time_aggregator import aggregate_by_time_window
from .metrics_calculator import compute_topic_tracking_metrics
from src.utils.file_io import save_dataframe, load_dataframe

class TopicTracker:
    """
    Stage 5: Topic Tracking
    Monitors historical temporal dynamics of topics across time windows:
    - Topic Volume
    - Growth Rate
    - Engagement
    - Sentiment Change
    - Hashtag Activity
    - Unique Users
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.time_window = self.config.get("time_window", "1D")
        self.engagement_weights = self.config.get("engagement_weights", {"likes": 1.0, "shares": 2.0, "comments": 1.5})
        self.output_path = self.config.get("output_path", "data/05_topic_tracking/tracked_topic_metrics.parquet")

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting Topic Tracking with time window '{self.time_window}'...")
        
        aggregated_df = aggregate_by_time_window(df, time_window=self.time_window)
        metrics_df = compute_topic_tracking_metrics(aggregated_df, self.engagement_weights)
        
        save_dataframe(metrics_df, self.output_path)
        logger.info(f"Topic Tracking completed. Saved {len(metrics_df)} metric rows to: {self.output_path}")
        return metrics_df

    def run_from_file(self, input_path: str) -> pd.DataFrame:
        df = load_dataframe(input_path)
        return self.run(df)
