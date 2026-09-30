import os
import pandas as pd
from typing import Tuple
from loguru import logger
from src.utils.file_io import save_dataframe, load_dataframe

class TimeBasedSplitter:
    """
    Stage 7: Time Based Data Splitting
    Partitions the dataset strictly chronologically to eliminate future data leakage:
    - Train set: First 70% of historical timeline
    - Validation set: Intermediate 20% of timeline
    - Test set: Final 10% of timeline
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.train_ratio = self.config.get("train_ratio", 0.70)
        self.valid_ratio = self.config.get("valid_ratio", 0.20)
        self.test_ratio = self.config.get("test_ratio", 0.10)
        self.splits_dir = self.config.get("splits_dir", "data/07_splits")

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Partitions the DataFrame along the time_bucket axis."""
        df_sorted = df.sort_values(by="time_bucket").reset_index(drop=True)
        unique_times = sorted(df_sorted["time_bucket"].unique())
        total_time_points = len(unique_times)
        
        logger.info(f"Total time points: {total_time_points}. Splitting 70-20-10 chronologically...")
        
        train_end_idx = int(total_time_points * self.train_ratio)
        valid_end_idx = int(total_time_points * (self.train_ratio + self.valid_ratio))
        
        train_cutoff = unique_times[train_end_idx]
        valid_cutoff = unique_times[valid_end_idx]
        
        train_df = df_sorted[df_sorted["time_bucket"] < train_cutoff].reset_index(drop=True)
        valid_df = df_sorted[
            (df_sorted["time_bucket"] >= train_cutoff) & (df_sorted["time_bucket"] < valid_cutoff)
        ].reset_index(drop=True)
        test_df = df_sorted[df_sorted["time_bucket"] >= valid_cutoff].reset_index(drop=True)
        
        logger.info("Chronological split completed:")
        logger.info(f"- Train (70%): {len(train_df)} samples (Timeline: {train_df['time_bucket'].min()} -> {train_df['time_bucket'].max()})")
        logger.info(f"- Valid (20%): {len(valid_df)} samples (Timeline: {valid_df['time_bucket'].min()} -> {valid_df['time_bucket'].max()})")
        logger.info(f"- Test (10%):  {len(test_df)} samples (Timeline: {test_df['time_bucket'].min()} -> {test_df['time_bucket'].max()})")
        
        train_path = os.path.join(self.splits_dir, "train.parquet")
        valid_path = os.path.join(self.splits_dir, "valid.parquet")
        test_path = os.path.join(self.splits_dir, "test.parquet")
        
        save_dataframe(train_df, train_path)
        save_dataframe(valid_df, valid_path)
        save_dataframe(test_df, test_path)
        
        return train_df, valid_df, test_df

    def split_from_file(self, input_path: str):
        df = load_dataframe(input_path)
        return self.split(df)
