"""
Automated Verification Suite for Two-Stage Cascade Architecture:
Tests Stage 1 (Classification Gate) and Stage 2 (Time-to-Trend Countdown Regression)
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_two_stage_cascade():
    print("=" * 70)
    print("RUNNING AUTOMATED VERIFICATION: TWO-STAGE CASCADE ARCHITECTURE")
    print("=" * 70)
    
    # Test 1: Verify Stage 6 outputs
    print("\n[TEST 1] Verifying Stage 6 Feature & Label Dataframe...")
    feat_path = "data/06_features_labels/features_and_labels.parquet"
    assert os.path.exists(feat_path), f"File missing: {feat_path}"
    df_feat = pd.read_parquet(feat_path)
    
    assert "is_trending" in df_feat.columns, "Missing 'is_trending' column"
    assert "time_to_trend_hours" in df_feat.columns, "Missing 'time_to_trend_hours' column"
    assert "time_to_trend_steps" in df_feat.columns, "Missing 'time_to_trend_steps' column"
    
    pos_mask = (df_feat["is_trending"] == 1)
    neg_mask = (df_feat["is_trending"] == 0)
    
    pos_count = int(pos_mask.sum())
    neg_count = int(neg_mask.sum())
    
    # In Hurdle model: positive instances must have non-null hours; negative instances must have NaN
    assert df_feat.loc[pos_mask, "time_to_trend_hours"].notna().all(), "All positive instances must have valid time_to_trend_hours!"
    assert df_feat.loc[neg_mask, "time_to_trend_hours"].isna().all(), "Negative instances must be NaN in Hurdle formulation!"
    print(f" -> PASS: Found {pos_count} positive trend instances with valid countdown ground-truth.")
    print(f" -> PASS: Found {neg_count} non-trending instances properly filtered (NaN).")
    
    # Test 2: Verify Stage 7 Split files
    print("\n[TEST 2] Verifying Stage 7 Chronological Splits...")
    for split_name in ["train", "valid", "test"]:
        p = f"data/07_splits/{split_name}.parquet"
        assert os.path.exists(p), f"Missing split file: {p}"
        sp_df = pd.read_parquet(p)
        assert "time_to_trend_hours" in sp_df.columns, f"Missing time_to_trend_hours in {split_name}"
        print(f" -> PASS: {split_name}.parquet has {len(sp_df)} rows and {sp_df['time_to_trend_hours'].notna().sum()} positive countdown targets.")

    # Test 3: Verify Stage 8 Prediction Output CSV & Cascade Columns
    print("\n[TEST 3] Verifying Stage 8 Cascade Output (predicted_trending_topics.csv)...")
    pred_path = "data/08_predictions/predicted_trending_topics.csv"
    assert os.path.exists(pred_path), f"Missing prediction file: {pred_path}"
    pred_df = pd.read_csv(pred_path)
    
    required_cascade_cols = [
        "predicted_trending",
        "trending_probability",
        "time_to_trend_hours",
        "time_to_trend_urgency",
        "projected_trend_onset",
        "cascade_stage",
        "topic_name"
    ]
    for col in required_cascade_cols:
        assert col in pred_df.columns, f"Missing cascade column '{col}' in predicted_trending_topics.csv"
        
    n_trending = len(pred_df)
    assert n_trending > 0, "No trending topics predicted!"
    
    # Ensure all predicted trending rows have valid hours and urgency
    assert pred_df["time_to_trend_hours"].notna().all(), "Predicted trending topics must have estimated hours!"
    assert (pred_df["time_to_trend_hours"] >= 3.0).all() and (pred_df["time_to_trend_hours"] <= 24.0).all(), "Predicted hours out of realistic bounds [3, 24]!"
    print(f" -> PASS: {n_trending} predicted trending topics all have valid Stage 2 countdown estimates.")
    print(" -> Sample Top 3 Trending Topics Countdown:")
    for idx, row in pred_df.head(3).iterrows():
        print(f"    * Topic: {row['topic_name']} | P(Trend): {row['trending_probability']:.1%} | Countdown: {row['time_to_trend_hours']}h | Urgency: {row['time_to_trend_urgency']} | Onset: {row['projected_trend_onset']}")

    # Test 4: Verify Evaluation Metrics JSON
    print("\n[TEST 4] Verifying Stage 2 Regression Metrics JSON...")
    metrics_path = "data/08_predictions/time_to_trend_metrics.json"
    assert os.path.exists(metrics_path), f"Missing metrics file: {metrics_path}"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    assert "mae_hours" in metrics, "Missing mae_hours in metrics"
    assert "accuracy_within_6h" in metrics, "Missing accuracy_within_6h in metrics"
    print(f" -> PASS: Stage 2 XGBRegressor MAE: {metrics['mae_hours']} hours.")
    print(f" -> PASS: Stage 2 Accuracy within ±6h: {metrics['accuracy_within_6h']:.1%}.")

    # Test 5: Verify Generated Visual Countdown Plot
    print("\n[TEST 5] Verifying Generated Countdown Plot...")
    plot_path = "data/08_predictions/plots/time_to_trend_countdown.png"
    assert os.path.exists(plot_path), f"Missing plot: {plot_path}"
    plot_size = os.path.getsize(plot_path)
    assert plot_size > 10000, "Plot image size too small!"
    print(f" -> PASS: time_to_trend_countdown.png generated successfully ({plot_size} bytes).")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! TWO-STAGE CASCADE IS FULLY OPERATIONAL.")
    print("=" * 70)

if __name__ == "__main__":
    test_two_stage_cascade()
