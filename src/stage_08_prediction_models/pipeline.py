import os
import json
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score
from .sarima_forecaster import SARIMAForecaster
from .feature_fusion import FeatureFusion
from .lightgbm_model import LightGBMTrendPredictor
from .evaluator import evaluate_trend_predictions
from .visualizer import (
    plot_comprehensive_evaluation_grid,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from src.utils.file_io import save_dataframe, load_dataframe, ensure_dir
from src.utils.metrics_utils import compute_mae, compute_rmse, compute_smape

class PredictionPipeline:
    """
    Stage 8 Pipeline:
    1. Topic Volume Forecasting (SARIMA)
    2. Feature Fusion (Social Listening Features + SARIMA Forecast Features)
    3. Trend Prediction (LightGBM)
    4. Model Evaluation & Visual Chart Generation (ROC, PR, Confusion Matrix, Feature Importance)
    5. Outputs ranked Predicted Trending Topics
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        sarima_cfg = self.config.get("sarima", {})
        lightgbm_cfg = self.config.get("lightgbm", {})
        
        self.forecaster = SARIMAForecaster(sarima_cfg)
        self.fusion = FeatureFusion()
        self.predictor = LightGBMTrendPredictor(lightgbm_cfg)
        
        self.predictions_dir = self.config.get("predictions_dir", "data/08_predictions")
        self.models_dir = self.config.get("models_dir", "models/lightgbm")

    def run(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> pd.DataFrame:
        logger.info("Starting Stage 8: Prediction Models & Feature Fusion...")
        
        # 1. Topic Volume Forecasting (SARIMA) on all 3 partitions
        logger.info("Generating SARIMA Forecast Features...")
        train_df_fc = self.forecaster.generate_forecast_features(train_df)
        valid_df_fc = self.forecaster.generate_forecast_features(valid_df)
        test_df_fc = self.forecaster.generate_forecast_features(test_df)
        
        # 2. Feature Fusion: Merge Social Listening + SARIMA features
        logger.info("Fusing features into Combined Feature Vector...")
        X_train, feature_cols = self.fusion.fuse(train_df_fc, is_training=True)
        y_train = train_df_fc["is_trending"].values
        
        X_valid, _ = self.fusion.fuse(valid_df_fc, is_training=False)
        y_valid = valid_df_fc["is_trending"].values
        
        X_test, _ = self.fusion.fuse(test_df_fc, is_training=False)
        y_test = test_df_fc["is_trending"].values
        
        # 3. Train Trend Prediction (LightGBM)
        self.predictor.train(X_train, y_train, X_valid, y_valid, feature_names=feature_cols)
        
        # 4. Tune decision threshold on Validation set (PDF Section 10)
        valid_probs = self.predictor.predict_proba(X_valid)
        best_threshold = 0.5
        best_val_f1 = -1.0
        for th in np.linspace(0.05, 0.75, 71):
            val_preds_th = (valid_probs >= th).astype(int)
            val_f1 = f1_score(y_valid, val_preds_th, zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_threshold = float(th)
        logger.info(f"Optimal classification threshold tuned on Validation set: {best_threshold:.3f} (Val F1: {best_val_f1:.4f})")
        
        # 5. Predict on Test set using the validation-tuned threshold
        test_probs = self.predictor.predict_proba(X_test)
        test_preds = (test_probs >= best_threshold).astype(int)
        
        # 6. Numerical Evaluation (Trend Prediction & SARIMA Forecasting per PDF Section 12)
        metrics = evaluate_trend_predictions(y_test, test_preds, test_probs)
        metrics["optimal_threshold"] = best_threshold
        metrics["val_f1"] = float(best_val_f1)
        
        if "future_volume" in test_df_fc.columns and "sarima_forecast_volume" in test_df_fc.columns:
            y_true_vol = test_df_fc["future_volume"].values
            y_pred_vol = test_df_fc["sarima_forecast_volume"].values
            sarima_metrics = {
                "sarima_mae": float(compute_mae(y_true_vol, y_pred_vol)),
                "sarima_rmse": float(compute_rmse(y_true_vol, y_pred_vol)),
                "sarima_smape": float(compute_smape(y_true_vol, y_pred_vol)),
            }
            metrics.update(sarima_metrics)
            logger.info("=== SARIMA TOPIC VOLUME FORECASTING EVALUATION (PDF Section 12) ===")
            logger.info(f"SARIMA MAE:   {sarima_metrics['sarima_mae']:.4f}")
            logger.info(f"SARIMA RMSE:  {sarima_metrics['sarima_rmse']:.4f}")
            logger.info(f"SARIMA sMAPE: {sarima_metrics['sarima_smape']:.2f}%")
        
        # 6. Visual Evaluation Charts
        plots_dir = os.path.join(self.predictions_dir, "plots")
        ensure_dir(os.path.join(plots_dir, ".gitkeep"))
        
        feature_importance_df = self.predictor.get_feature_importance()
        
        try:
            logger.info("Generating visual model evaluation plots...")
            plot_comprehensive_evaluation_grid(
                y_test,
                test_preds,
                test_probs,
                feature_importance_df=feature_importance_df,
                save_path=os.path.join(plots_dir, "evaluation_dashboard.png")
            )
            plot_confusion_matrix(
                y_test,
                test_preds,
                save_path=os.path.join(plots_dir, "confusion_matrix.png")
            )
            plot_roc_curve(
                y_test,
                test_probs,
                save_path=os.path.join(plots_dir, "roc_curve.png")
            )
            plot_precision_recall_curve(
                y_test,
                test_probs,
                save_path=os.path.join(plots_dir, "precision_recall_curve.png")
            )
            plot_feature_importance(
                feature_importance_df,
                save_path=os.path.join(plots_dir, "feature_importance.png")
            )
            logger.info(f"Visual evaluation plots successfully saved to: {plots_dir}")
        except Exception as e:
            logger.warning(f"Could not generate visual plots: {e}")

        # 7. Generate Predicted Trending Topics
        results_df = test_df_fc.copy()
        results_df["predicted_trending"] = test_preds
        results_df["trending_probability"] = test_probs
        
        # Save prediction results
        ensure_dir(os.path.join(self.predictions_dir, "predicted_trending_topics.csv"))
        save_dataframe(results_df, os.path.join(self.predictions_dir, "all_test_predictions.parquet"))
        
        trending_topics = results_df[results_df["predicted_trending"] == 1].sort_values(
            by="trending_probability", ascending=False
        )
        save_dataframe(trending_topics, os.path.join(self.predictions_dir, "predicted_trending_topics.csv"))
        
        # Save model and evaluation metrics
        model_path = os.path.join(self.models_dir, "lightgbm_trend_model.joblib")
        self.predictor.save(model_path)
        
        metrics_path = os.path.join(self.predictions_dir, "evaluation_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"Stage 8 completed! Ranked trending topics saved to: {os.path.join(self.predictions_dir, 'predicted_trending_topics.csv')}")
        return trending_topics

    def run_from_splits(self, splits_dir: str = "data/07_splits"):
        train_df = load_dataframe(os.path.join(splits_dir, "train.parquet"))
        valid_df = load_dataframe(os.path.join(splits_dir, "valid.parquet"))
        test_df = load_dataframe(os.path.join(splits_dir, "test.parquet"))
        return self.run(train_df, valid_df, test_df)
