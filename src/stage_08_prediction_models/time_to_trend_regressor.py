"""
Two-Stage Cascade Architecture: Stage 2 Time-to-Trend Regression Model
Predicts the exact countdown time (hours/steps) until trend onset for topics
confirmed as trending by the Stage 1 Classifier.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
from loguru import logger
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.utils.file_io import ensure_dir

class TimeToTrendRegressor:
    """
    Two-Stage Cascade Architecture:
    - Stage 1 (Gatekeeper): Binary Classifier (XGBoost/LightGBM) determines whether a topic will trend.
    - Stage 2 (Estimator): Regressor predicts how many hours until the topic reaches trend onset.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.step_hours = float(self.config.get("step_hours", 6.0))
        self.n_estimators = int(self.config.get("n_estimators", 100))
        self.learning_rate = float(self.config.get("learning_rate", 0.05))
        self.max_depth = int(self.config.get("max_depth", 3))
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_names = None
        self.metrics = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train_hours: np.ndarray,
        y_train_trending: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None
    ):
        """
        Fits the Stage 2 Regressor exclusively on positive trending instances (Hurdle formulation).
        """
        self.feature_names = feature_names
        
        # Mask strictly positive trending instances with valid onset ground-truth
        mask = ~np.isnan(y_train_hours) & (y_train_hours > 0)
        if y_train_trending is not None:
            mask = mask & (y_train_trending == 1)
            
        n_pos = int(np.sum(mask))
        if n_pos < 3:
            logger.warning(f"Insufficient positive instances for Time-to-Trend regressor (found {n_pos}). Using heuristic baseline.")
            self.is_fitted = False
            return
            
        X_sub = X_train[mask]
        y_sub = y_train_hours[mask]
        
        logger.info(f"[Two-Stage Cascade - Stage 2] Training Time-to-Trend XGBRegressor on {n_pos} positive trend instances...")
        self.model.fit(X_sub, y_sub)
        self.is_fitted = True
        logger.info("[Two-Stage Cascade - Stage 2] Regressor training complete.")

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test_hours: np.ndarray,
        y_test_trending: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluates Stage 2 regression accuracy (MAE, RMSE, accuracy within 1 time-bucket).
        """
        mask = ~np.isnan(y_test_hours) & (y_test_hours > 0)
        if y_test_trending is not None:
            mask = mask & (y_test_trending == 1)
            
        n_eval = int(np.sum(mask))
        if n_eval == 0 or not self.is_fitted:
            logger.warning("No ground-truth positive instances in test set for Time-to-Trend evaluation.")
            return {"mae": 0.0, "rmse": 0.0, "accuracy_within_6h": 0.0, "test_samples": 0}
            
        X_sub = X_test[mask]
        y_sub = y_test_hours[mask]
        preds = self.model.predict(X_sub)
        preds = np.clip(preds, 3.0, 24.0)
        
        mae = float(mean_absolute_error(y_sub, preds))
        rmse = float(np.sqrt(mean_squared_error(y_sub, preds)))
        acc_6h = float(np.mean(np.abs(preds - y_sub) <= self.step_hours))
        
        self.metrics = {
            "mae_hours": round(mae, 2),
            "rmse_hours": round(rmse, 2),
            "accuracy_within_6h": round(acc_6h, 4),
            "test_positive_samples": n_eval,
            "mean_ground_truth_hours": round(float(np.mean(y_sub)), 1),
            "mean_predicted_hours": round(float(np.mean(preds)), 1)
        }
        
        logger.info(f"[Two-Stage Cascade - Stage 2 Results] MAE: {mae:.2f}h | RMSE: {rmse:.2f}h | Accuracy (±6h): {acc_6h:.1%}")
        return self.metrics

    def predict_cascade(
        self,
        test_df: pd.DataFrame,
        X_test: np.ndarray,
        predicted_trending: np.ndarray,
        probabilities: np.ndarray,
        threshold: float = 0.43
    ) -> pd.DataFrame:
        """
        Executes cascade inference:
        - Topics where predicted_trending == 1 (or prob >= threshold) pass Stage 1 and receive Stage 2 countdown.
        - Non-trending topics are filtered out (hours = NaN, urgency = Normal).
        """
        df = test_df.copy()
        n = len(df)
        
        time_to_trend_hours = np.full(n, np.nan)
        time_to_trend_urgency = ["Normal (Not Trending)"] * n
        projected_onset = ["N/A"] * n
        cascade_stage = ["Stage 1: Filtered Out (Normal)"] * n
        
        # Determine candidates that pass Stage 1
        pos_mask = (predicted_trending == 1) | (probabilities >= threshold)
        
        if np.any(pos_mask) and self.is_fitted:
            X_pos = X_test[pos_mask]
            raw_hours = self.model.predict(X_pos)
            
            # Incorporate probability momentum scaling: topics with higher probability/acceleration erupt sooner
            probs_pos = probabilities[pos_mask]
            momentum_adjustment = 1.0 - 0.25 * np.clip((probs_pos - threshold) / max(1.0 - threshold, 0.1), 0.0, 1.0)
            adjusted_hours = np.clip(raw_hours * momentum_adjustment, 3.0, 24.0)
            
            # Map back to positions
            pos_indices = np.where(pos_mask)[0]
            for idx, h, p in zip(pos_indices, adjusted_hours, probs_pos):
                h_round = round(float(h), 1)
                time_to_trend_hours[idx] = h_round
                cascade_stage[idx] = "Stage 1: Passed -> Stage 2: Countdown Estimated"
                
                # Determine Urgency Tier
                if h_round <= 6.0:
                    time_to_trend_urgency[idx] = "Imminent (< 6h) - Critical Warning"
                elif h_round <= 12.0:
                    time_to_trend_urgency[idx] = "Fast Approaching (6h - 12h)"
                elif h_round <= 18.0:
                    time_to_trend_urgency[idx] = "Developing (12h - 18h)"
                else:
                    time_to_trend_urgency[idx] = "Early Horizon (> 18h)"
                    
                # Compute projected timestamp if time_bucket is present
                if "time_bucket" in df.columns:
                    try:
                        base_t = pd.to_datetime(df.iloc[idx]["time_bucket"])
                        onset_dt = base_t + pd.to_timedelta(h_round, unit="h")
                        projected_onset[idx] = onset_dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        projected_onset[idx] = f"+{h_round}h"
                        
        df["time_to_trend_hours"] = time_to_trend_hours
        df["time_to_trend_urgency"] = time_to_trend_urgency
        df["projected_trend_onset"] = projected_onset
        df["cascade_stage"] = cascade_stage
        
        return df

    def plot_time_to_trend_countdown(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Visualizes the Two-Stage Cascade countdown distribution and urgency tiers.
        """
        trending_subset = results_df[results_df["predicted_trending"] == 1].copy()
        if trending_subset.empty or "time_to_trend_hours" not in trending_subset.columns:
            logger.warning("No predicted trending topics found to plot Time-to-Trend distribution.")
            return None, None
            
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        
        # Panel 1: Urgency Tier Distribution
        ax1 = axes[0]
        urgency_counts = trending_subset["time_to_trend_urgency"].value_counts()
        
        palette_map = {
            "Imminent (< 6h) - Critical Warning": "#d62728",
            "Fast Approaching (6h - 12h)": "#ff7f0e",
            "Developing (12h - 18h)": "#1f77b4",
            "Early Horizon (> 18h)": "#2ca02c",
            "Normal (Not Trending)": "#7f7f7f"
        }
        bar_colors = [palette_map.get(u, "#333333") for u in urgency_counts.index]
        
        bars = ax1.bar(urgency_counts.index, urgency_counts.values, color=bar_colors, alpha=0.9, edgecolor="black", lw=0.8)
        for b in bars:
            h = b.get_height()
            ax1.annotate(f"{int(h)}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontweight="bold")
                         
        ax1.set_title("Stage 2: Time-to-Trend Urgency Breakdown", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Number of Trending Topics", fontsize=11)
        ax1.set_xticklabels(urgency_counts.index, rotation=18, ha="right", fontsize=9.5)
        ax1.grid(True, linestyle="--", alpha=0.5)
        
        # Panel 2: Probability vs Countdown Hours Scatter
        ax2 = axes[1]
        probs = trending_subset["trending_probability"].values
        hours = trending_subset["time_to_trend_hours"].values
        scatter_colors = [palette_map.get(u, "#1f77b4") for u in trending_subset["time_to_trend_urgency"]]
        
        scatter = ax2.scatter(hours, probs, c=scatter_colors, s=75, alpha=0.85, edgecolors="white", linewidths=1.2, zorder=4)
        
        # Decision boundary lines
        ax2.axvline(x=6.0, color="#d62728", linestyle=":", lw=1.5, label="6h Boundary (1 Step)")
        ax2.axvline(x=12.0, color="#ff7f0e", linestyle=":", lw=1.5, label="12h Boundary (2 Steps)")
        ax2.axvline(x=18.0, color="#1f77b4", linestyle=":", lw=1.5, label="18h Boundary (3 Steps)")
        
        ax2.set_title("Trending Probability vs. Projected Hours to Onset", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Estimated Hours to Trend Onset (h)", fontsize=11)
        ax2.set_ylabel("Stage 1 Trending Probability P(Trend)", fontsize=11)
        ax2.set_xlim([2.0, 25.0])
        ax2.set_ylim([0.35, 1.0])
        ax2.legend(loc="upper right", frameon=True, fontsize=9.5)
        ax2.grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(save_path)
            try:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved Time-to-Trend Countdown Plot to: {save_path}")
            except Exception as e:
                logger.warning(f"Could not save countdown plot: {e}")
                
        return fig, axes
