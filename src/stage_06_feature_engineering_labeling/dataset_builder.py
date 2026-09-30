import pandas as pd
from loguru import logger
from .feature_extractor import extract_social_features
from .labeler import generate_trending_labels
from src.utils.file_io import save_dataframe, load_dataframe

class FeatureAndLabelBuilder:
    """
    Stage 6: Feature Engineering & Label Data
    Builds the tabular dataset containing engineered features and future ground-truth Trending labels.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        labeling_cfg = self.config.get("labeling", {})
        self.future_window_steps = labeling_cfg.get("future_time_window_steps", 3)
        self.future_growth_threshold = labeling_cfg.get("future_growth_threshold", 0.5)
        self.future_volume_threshold = labeling_cfg.get("future_volume_threshold", 20.0)
        self.output_path = self.config.get("output_path", "data/06_features_labels/features_and_labels.parquet")

    def run(self, tracked_metrics_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting Stage 6: Feature Engineering & Labeling...")
        
        # Branch 1: Feature Engineering
        featured_df = extract_social_features(tracked_metrics_df)
        
        # Branch 2: Label Data
        labeled_df = generate_trending_labels(
            featured_df,
            future_window_steps=self.future_window_steps,
            future_growth_threshold=self.future_growth_threshold,
            future_volume_threshold=self.future_volume_threshold
        )
        
        trending_count = (labeled_df["is_trending"] == 1).sum()
        total_count = len(labeled_df)
        trending_pct = trending_count / max(total_count, 1)
        logger.info(f"Dataset creation complete. Total samples: {total_count}. Trending: {trending_count} ({trending_pct:.1%})")
        
        save_dataframe(labeled_df, self.output_path)
        return labeled_df

    def run_from_file(self, input_path: str) -> pd.DataFrame:
        df = load_dataframe(input_path)
        return self.run(df)
