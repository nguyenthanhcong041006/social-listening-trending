import pandas as pd
from loguru import logger
from typing import List

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Step 1: Remove Duplicate
    Removes exact duplicate posts or identical text content.
    """
    initial_count = len(df)
    if subset is None:
        subset = ["text"]
    
    df_dedup = df.drop_duplicates(subset=subset).reset_index(drop=True)
    removed_count = initial_count - len(df_dedup)
    logger.info(f"[Remove Duplicate] Removed {removed_count} duplicate posts. Remaining: {len(df_dedup)} posts.")
    return df_dedup
