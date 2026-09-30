import pandas as pd
import numpy as np

def generate_trending_labels(
    df: pd.DataFrame,
    future_window_steps: int = 3,
    future_growth_threshold: float = 0.5,
    future_volume_threshold: float = 20.0
) -> pd.DataFrame:
    """
    Sub-stage 6.2: Label Data (Ground Truth Generation)
    Generates binary ground-truth labels: Trending (1) or Not Trending (0) based on future performance:
    - Future Time Window: Forward lookahead over `future_window_steps` (t + 1 to t + k)
    - Future Volume: Sum of topic post volume across the forward window
    - Future Growth: Volume growth rate comparing forward window to current volume baseline
    
    Labeling Decision Rule:
    Trending = 1 if:
        (Future Growth >= future_growth_threshold) AND (Future Volume >= future_volume_threshold)
    Otherwise:
        Not Trending = 0
    """
    df = df.sort_values(by=["topic_id", "time_bucket"]).copy()
    
    # Calculate Future Volume across the forward time window
    future_vols = []
    for topic_id, group in df.groupby("topic_id"):
        vols = group["topic_volume"].values
        f_vol = np.zeros(len(vols))
        for i in range(len(vols)):
            future_slice = vols[i + 1 : i + 1 + future_window_steps]
            f_vol[i] = np.sum(future_slice) if len(future_slice) > 0 else np.nan
        group = group.copy()
        group["future_volume"] = f_vol
        future_vols.append(group)
        
    df_labeled = pd.concat(future_vols).sort_values(by=["topic_id", "time_bucket"]).reset_index(drop=True)
    
    # Calculate Future Growth relative to current volume scaled over the window
    current_vol_normalized = df_labeled["topic_volume"] * future_window_steps
    df_labeled["future_growth"] = np.where(
        current_vol_normalized > 0,
        (df_labeled["future_volume"] - current_vol_normalized) / current_vol_normalized,
        0.0
    )
    
    # Apply threshold condition
    is_trending = (
        (df_labeled["future_growth"] >= future_growth_threshold) &
        (df_labeled["future_volume"] >= future_volume_threshold)
    ).astype(int)
    
    df_labeled["is_trending"] = is_trending
    
    # Drop trailing rows lacking complete future lookahead horizon
    df_clean_labeled = df_labeled.dropna(subset=["future_volume"]).reset_index(drop=True)
    return df_clean_labeled
