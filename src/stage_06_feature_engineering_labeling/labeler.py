import pandas as pd
import numpy as np

def generate_trending_labels(
    df: pd.DataFrame,
    future_window_steps: int = 3,
    future_growth_threshold: float = 0.5,
    future_volume_threshold: float = 5.0,
    step_hours: float = 6.0
) -> pd.DataFrame:
    """
    Sub-stage 6.2: Label Data (Ground Truth Generation for Two-Stage Cascade)
    
    Stage 1 Target: Binary Classification
    - is_trending: 1 if (Future Growth >= future_growth_threshold) AND (Future Volume >= future_volume_threshold), else 0.
    
    Stage 2 Target: Time-to-Trend Regression (Hurdle Formulation)
    - time_to_trend_steps: Forward steps until trend onset/burst (1, 2, or 3) for positive instances; NaN for non-trending.
    - time_to_trend_hours: Exact countdown time in hours (steps * step_hours, e.g. 6.0, 12.0, 18.0h) for positive instances.
    """
    df = df.sort_values(by=["topic_id", "time_bucket"]).copy()
    
    # Calculate Future Volume and Time-to-Trend across the forward time window
    future_vols = []
    for topic_id, group in df.groupby("topic_id"):
        vols = group["topic_volume"].values
        n_samples = len(vols)
        f_vol = np.zeros(n_samples)
        onset_steps = np.full(n_samples, np.nan)
        onset_hours = np.full(n_samples, np.nan)
        
        for i in range(n_samples):
            future_slice = vols[i + 1 : i + 1 + future_window_steps]
            if len(future_slice) > 0:
                f_vol[i] = np.sum(future_slice)
                
                # Determine earliest onset step within the forward window
                curr_v = vols[i]
                found_onset = False
                for s_idx, forward_v in enumerate(future_slice):
                    s = s_idx + 1
                    cum_vol_s = np.sum(future_slice[:s])
                    base_norm_s = curr_v * s
                    cum_growth_s = (cum_vol_s - base_norm_s) / max(base_norm_s, 1.0) if base_norm_s > 0 else 0.0
                    
                    if cum_growth_s >= future_growth_threshold and cum_vol_s >= future_volume_threshold:
                        onset_steps[i] = float(s)
                        onset_hours[i] = float(s * step_hours)
                        found_onset = True
                        break
                        
                if not found_onset:
                    # Fallback to the peak volume step or the window boundary
                    peak_step = int(np.argmax(future_slice) + 1)
                    onset_steps[i] = float(peak_step)
                    onset_hours[i] = float(peak_step * step_hours)
            else:
                f_vol[i] = np.nan
                
        group = group.copy()
        group["future_volume"] = f_vol
        group["_candidate_onset_steps"] = onset_steps
        group["_candidate_onset_hours"] = onset_hours
        future_vols.append(group)
        
    df_labeled = pd.concat(future_vols).sort_values(by=["topic_id", "time_bucket"]).reset_index(drop=True)
    
    # Calculate Future Growth relative to current volume scaled over the window
    current_vol_normalized = df_labeled["topic_volume"] * future_window_steps
    df_labeled["future_growth"] = np.where(
        current_vol_normalized > 0,
        (df_labeled["future_volume"] - current_vol_normalized) / current_vol_normalized,
        0.0
    )
    
    # Apply threshold condition for Stage 1 Binary Label
    is_trending = (
        (df_labeled["future_growth"] >= future_growth_threshold) &
        (df_labeled["future_volume"] >= future_volume_threshold)
    ).astype(int)
    
    df_labeled["is_trending"] = is_trending
    
    # Stage 2 Target: Hurdle selection (only positive trending instances have valid countdown values)
    df_labeled["time_to_trend_steps"] = np.where(
        df_labeled["is_trending"] == 1,
        df_labeled["_candidate_onset_steps"],
        np.nan
    )
    df_labeled["time_to_trend_hours"] = np.where(
        df_labeled["is_trending"] == 1,
        df_labeled["_candidate_onset_hours"],
        np.nan
    )
    df_labeled.drop(columns=["_candidate_onset_steps", "_candidate_onset_hours"], inplace=True)
    
    # Drop trailing rows lacking complete future lookahead horizon
    df_clean_labeled = df_labeled.dropna(subset=["future_volume"]).reset_index(drop=True)
    return df_clean_labeled
