import os
import json
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score
from typing import Dict, Any, Tuple, List

from .sarima_forecaster import SARIMAForecaster
from .sarimax_forecaster import SARIMAXForecaster
from .feature_fusion import FeatureFusion
from .lightgbm_model import LightGBMTrendPredictor
from .xgboost_model import XGBoostTrendPredictor
from .evaluator import evaluate_trend_predictions
from .visualizer import (
    plot_comprehensive_evaluation_grid,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_learning_curves,
    plot_calibration_curve,
    plot_threshold_optimization,
    plot_trend_decay_curves,
    plot_ablation_study,
    plot_model_comparison_roc,
    plot_model_comparison_pr,
    plot_model_comparison_barchart,
    plot_model_comparison_radar,
    plot_sarima_vs_sarimax_comparison,
    plot_model_comparison_confusion_matrices,
    plot_model_comparison_calibration,
    export_model_comparison_table,
)
from .trend_duration import estimate_trend_duration_and_persistence
from src.utils.file_io import save_dataframe, load_dataframe, ensure_dir
from src.utils.metrics_utils import compute_mae, compute_rmse, compute_smape

class PredictionPipeline:
    """
    Stage 8 Pipeline:
    1. Topic Volume Forecasting (SARIMA & SARIMAX with Exogenous covariates)
    2. Feature Fusion (Social Listening Features + Forecast Features)
    3. Multi-Model Hybrid Trend Prediction & Comparison:
       - SARIMA + LightGBM (Baseline)
       - SARIMAX + LightGBM
       - SARIMA + XGBoost
       - SARIMAX + XGBoost
    4. Comprehensive Multi-Model Benchmarking & Evaluation (Macro-F1, MCC, ROC-AUC, PR-AUC, etc.)
    5. Generation of Publication-Grade Research Plots & LaTeX Summary Table
    6. Ranked Emerging Trend Predictions & 24h Persistence Estimation
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        sarima_cfg = self.config.get("sarima", {})
        sarimax_cfg = self.config.get("sarimax", sarima_cfg)
        lightgbm_cfg = self.config.get("lightgbm", {})
        xgboost_cfg = self.config.get("xgboost", {})
        
        self.forecaster_sarima = SARIMAForecaster(sarima_cfg)
        self.forecaster_sarimax = SARIMAXForecaster(sarimax_cfg)
        self.fusion = FeatureFusion()
        
        self.lgb_config = lightgbm_cfg
        self.xgb_config = xgboost_cfg
        
        self.predictions_dir = self.config.get("predictions_dir", "data/08_predictions")
        self.models_dir = self.config.get("models_dir", "models")
        self.compare_models = self.config.get("compare_models", True)

    def _optimize_threshold(self, predictor, X_valid: np.ndarray, y_valid: np.ndarray) -> Tuple[float, float]:
        """Tuning decision boundary threshold on validation set to optimize F1 score."""
        val_probs = predictor.predict_proba(X_valid)
        best_threshold = 0.5
        best_val_f1 = -1.0
        for th in np.linspace(0.05, 0.85, 81):
            val_preds = (val_probs >= th).astype(int)
            val_f1 = f1_score(y_valid, val_preds, zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = float(val_f1)
                best_threshold = float(th)
        return best_threshold, best_val_f1

    def run_model_comparison(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
        """
        Executes a rigorous 4-way benchmark comparing:
        1. SARIMA + LightGBM
        2. SARIMAX + LightGBM
        3. SARIMA + XGBoost
        4. SARIMAX + XGBoost
        """
        logger.info("==========================================================================")
        logger.info("STARTING MULTI-MODEL RESEARCH BENCHMARK: SARIMA/SARIMAX + LightGBM/XGBoost")
        logger.info("==========================================================================")

        plots_dir = os.path.join(self.predictions_dir, "plots")
        comparison_plots_dir = os.path.join(plots_dir, "comparison")
        ensure_dir(os.path.join(comparison_plots_dir, ".gitkeep"))

        # Step 1: Generate Forecast Features for both SARIMA and SARIMAX
        logger.info("Generating SARIMA Univariate Forecast Features...")
        train_sarima = self.forecaster_sarima.generate_forecast_features(train_df)
        valid_sarima = self.forecaster_sarima.generate_forecast_features(valid_df)
        test_sarima = self.forecaster_sarima.generate_forecast_features(test_df)

        logger.info("Generating SARIMAX Exogenous Forecast Features...")
        train_sarimax = self.forecaster_sarimax.generate_forecast_features(train_df)
        valid_sarimax = self.forecaster_sarimax.generate_forecast_features(valid_df)
        test_sarimax = self.forecaster_sarimax.generate_forecast_features(test_df)

        # Step 2: Evaluate Time-Series Volume Forecasting (SARIMA vs SARIMAX)
        ts_comparison = []
        if "future_volume" in test_df.columns:
            y_true_vol = test_df["future_volume"].values
            
            # SARIMA metrics
            y_pred_sarima = test_sarima["sarima_forecast_volume"].values
            sarima_mae = float(compute_mae(y_true_vol, y_pred_sarima))
            sarima_rmse = float(compute_rmse(y_true_vol, y_pred_sarima))
            sarima_smape = float(compute_smape(y_true_vol, y_pred_sarima))
            ts_comparison.append({
                "forecaster": "SARIMA (Univariate)",
                "mae": sarima_mae,
                "rmse": sarima_rmse,
                "smape": sarima_smape
            })

            # SARIMAX metrics
            y_pred_sarimax = test_sarimax["sarimax_forecast_volume"].values
            sarimax_mae = float(compute_mae(y_true_vol, y_pred_sarimax))
            sarimax_rmse = float(compute_rmse(y_true_vol, y_pred_sarimax))
            sarimax_smape = float(compute_smape(y_true_vol, y_pred_sarimax))
            ts_comparison.append({
                "forecaster": "SARIMAX (Exogenous)",
                "mae": sarimax_mae,
                "rmse": sarimax_rmse,
                "smape": sarimax_smape
            })

            ts_df = pd.DataFrame(ts_comparison)
            logger.info("=== TIME-SERIES VOLUME FORECASTING COMPARISON ===")
            logger.info(f"\n{ts_df.to_string(index=False)}")
            
            # Plot Time-Series Comparison
            try:
                plot_sarima_vs_sarimax_comparison(
                    ts_df,
                    save_path=os.path.join(comparison_plots_dir, "sarima_vs_sarimax_forecasting.png")
                )
            except Exception as e:
                logger.warning(f"Could not plot SARIMA vs SARIMAX forecasting: {e}")

        # Step 3: Feature Fusion for SARIMA and SARIMAX sets
        logger.info("Fusing features for SARIMA and SARIMAX pipelines...")
        X_train_sarima, feat_cols_sarima = self.fusion.fuse(train_sarima, forecaster_type="sarima", is_training=True)
        X_valid_sarima, _ = self.fusion.fuse(valid_sarima, forecaster_type="sarima", is_training=False)
        X_test_sarima, _ = self.fusion.fuse(test_sarima, forecaster_type="sarima", is_training=False)
        y_train = train_df["is_trending"].values
        y_valid = valid_df["is_trending"].values
        y_test = test_df["is_trending"].values

        X_train_sarimax, feat_cols_sarimax = self.fusion.fuse(train_sarimax, forecaster_type="sarimax", is_training=True)
        X_valid_sarimax, _ = self.fusion.fuse(valid_sarimax, forecaster_type="sarimax", is_training=False)
        X_test_sarimax, _ = self.fusion.fuse(test_sarimax, forecaster_type="sarimax", is_training=False)

        # Step 4: Define the 4 model combinations
        model_combos = [
            {
                "name": "SARIMA + LightGBM",
                "classifier_cls": LightGBMTrendPredictor,
                "classifier_cfg": self.lgb_config,
                "X_train": X_train_sarima,
                "X_valid": X_valid_sarima,
                "X_test": X_test_sarima,
                "feature_names": feat_cols_sarima,
                "test_df": test_sarima,
                "save_dir": os.path.join(self.models_dir, "lightgbm")
            },
            {
                "name": "SARIMAX + LightGBM",
                "classifier_cls": LightGBMTrendPredictor,
                "classifier_cfg": self.lgb_config,
                "X_train": X_train_sarimax,
                "X_valid": X_valid_sarimax,
                "X_test": X_test_sarimax,
                "feature_names": feat_cols_sarimax,
                "test_df": test_sarimax,
                "save_dir": os.path.join(self.models_dir, "lightgbm")
            },
            {
                "name": "SARIMA + XGBoost",
                "classifier_cls": XGBoostTrendPredictor,
                "classifier_cfg": self.xgb_config,
                "X_train": X_train_sarima,
                "X_valid": X_valid_sarima,
                "X_test": X_test_sarima,
                "feature_names": feat_cols_sarima,
                "test_df": test_sarima,
                "save_dir": os.path.join(self.models_dir, "xgboost")
            },
            {
                "name": "SARIMAX + XGBoost",
                "classifier_cls": XGBoostTrendPredictor,
                "classifier_cfg": self.xgb_config,
                "X_train": X_train_sarimax,
                "X_valid": X_valid_sarimax,
                "X_test": X_test_sarimax,
                "feature_names": feat_cols_sarimax,
                "test_df": test_sarimax,
                "save_dir": os.path.join(self.models_dir, "xgboost")
            },
        ]

        # Step 5: Train and evaluate each model combination
        comparison_records = []
        models_probs = {}
        models_preds = {}
        trained_models = {}
        best_thresholds = {}

        for combo in model_combos:
            m_name = combo["name"]
            logger.info(f"\n--- Training & Evaluating: [{m_name}] ---")
            predictor = combo["classifier_cls"](combo["classifier_cfg"])

            predictor.train(
                combo["X_train"],
                y_train,
                X_valid=combo["X_valid"],
                y_valid=y_valid,
                X_test=combo["X_test"],
                y_test=y_test,
                feature_names=combo["feature_names"]
            )

            # Threshold tuning on Validation set
            best_th, best_val_f1 = self._optimize_threshold(predictor, combo["X_valid"], y_valid)
            best_thresholds[m_name] = best_th
            logger.info(f"[{m_name}] Optimal Threshold tuned on Val: θ*={best_th:.3f} (Val F1: {best_val_f1:.4f})")

            # Inference on Test set
            test_prob = predictor.predict_proba(combo["X_test"])
            test_pred = (test_prob >= best_th).astype(int)

            models_probs[m_name] = test_prob
            models_preds[m_name] = test_pred
            trained_models[m_name] = predictor

            # Academic Metrics Evaluation
            metrics = evaluate_trend_predictions(y_test, test_pred, test_prob, model_name=m_name)
            metrics["optimal_threshold"] = best_th
            metrics["val_f1"] = best_val_f1
            metrics["best_iteration"] = int(getattr(predictor, "best_iteration", 0))
            comparison_records.append(metrics)

            # Save individual model
            safe_filename = m_name.lower().replace(" ", "_").replace("+", "plus") + ".joblib"
            save_path = os.path.join(combo["save_dir"], safe_filename)
            predictor.save(save_path)

        comparison_df = pd.DataFrame(comparison_records)
        logger.info("\n==========================================================================")
        logger.info("FINAL MULTI-MODEL RESEARCH BENCHMARK SUMMARY (TEST SET)")
        logger.info("==========================================================================")
        logger.info(f"\n{comparison_df[['model_name', 'macro_f1', 'f1_score', 'precision', 'recall', 'roc_auc', 'pr_auc', 'mcc', 'optimal_threshold']].to_string(index=False)}")

        # Step 6: Generate Multi-Model Comparison Visualizations
        logger.info("Generating multi-model comparative research plots...")
        try:
            plot_model_comparison_roc(
                models_probs,
                y_test,
                save_path=os.path.join(comparison_plots_dir, "model_comparison_roc.png")
            )
            plot_model_comparison_pr(
                models_probs,
                y_test,
                save_path=os.path.join(comparison_plots_dir, "model_comparison_pr.png")
            )
            plot_model_comparison_barchart(
                comparison_df,
                save_path=os.path.join(comparison_plots_dir, "model_comparison_metrics_barchart.png")
            )
            plot_model_comparison_radar(
                comparison_df,
                save_path=os.path.join(comparison_plots_dir, "model_comparison_radar.png")
            )
            plot_model_comparison_confusion_matrices(
                models_preds,
                y_test,
                save_path=os.path.join(comparison_plots_dir, "model_comparison_confusion_matrices.png")
            )
            plot_model_comparison_calibration(
                models_probs,
                y_test,
                save_path=os.path.join(comparison_plots_dir, "model_comparison_calibration.png")
            )
            export_model_comparison_table(
                comparison_df,
                save_dir=self.predictions_dir
            )

            # Automatically sync comparison plots and table image to docs/images for the research report
            docs_images_dir = self.config.get("docs_images_dir", "docs/images")
            try:
                import shutil
                ensure_dir(os.path.join(docs_images_dir, ".gitkeep"))
                if os.path.exists(comparison_plots_dir):
                    for fname in os.listdir(comparison_plots_dir):
                        if fname.endswith(".png"):
                            shutil.copy2(os.path.join(comparison_plots_dir, fname), os.path.join(docs_images_dir, fname))
                table_img = os.path.join(self.predictions_dir, "model_comparison_table.png")
                if os.path.exists(table_img):
                    shutil.copy2(table_img, os.path.join(docs_images_dir, "model_comparison_table.png"))
                logger.info(f"Comparison images successfully synced to research docs folder: {docs_images_dir}")
            except Exception as e:
                logger.warning(f"Could not sync comparison images to {docs_images_dir}: {e}")

            logger.info("All model comparison plots and LaTeX tables generated successfully!")
        except Exception as e:
            logger.error(f"Error during comparative plot generation: {e}")

        # Step 7: Select Best Performing Model for downstream trend prediction & persistence
        # Ranked by Macro-F1 + ROC-AUC + MCC
        comparison_df["composite_score"] = (
            comparison_df["macro_f1"] * 0.4 +
            comparison_df["mcc"] * 0.3 +
            comparison_df["roc_auc"] * 0.3
        )
        best_row = comparison_df.sort_values(by="composite_score", ascending=False).iloc[0]
        best_model_name = best_row["model_name"]
        logger.info(f"Top-Ranked Model across research criteria: [{best_model_name}] (Composite Score: {best_row['composite_score']:.4f})")

        # Determine which test dataframe to use for predictions
        best_combo = next(c for c in model_combos if c["name"] == best_model_name)
        best_test_df = best_combo["test_df"].copy()
        best_predictor = trained_models[best_model_name]
        best_th = best_thresholds[best_model_name]
        best_prob = models_probs[best_model_name]
        best_pred = models_preds[best_model_name]

        best_test_df["predicted_trending"] = best_pred
        best_test_df["trending_probability"] = best_prob
        best_test_df["best_model_used"] = best_model_name

        # Estimate Trend Lifespan & 24h Persistence
        best_test_df = estimate_trend_duration_and_persistence(best_test_df, threshold=best_th)

        # Standard plots for the primary/best model
        try:
            feat_imp = best_predictor.get_feature_importance()
            plot_comprehensive_evaluation_grid(
                y_test, best_pred, best_prob,
                feature_importance_df=feat_imp,
                save_path=os.path.join(plots_dir, "evaluation_dashboard.png")
            )
            plot_feature_importance(
                feat_imp,
                save_path=os.path.join(plots_dir, "feature_importance.png")
            )
            plot_learning_curves(
                best_predictor.evals_result,
                best_iteration=getattr(best_predictor, "best_iteration", 0),
                save_path=os.path.join(plots_dir, "learning_curves.png")
            )
            plot_threshold_optimization(
                y_test, best_prob,
                optimal_threshold=best_th,
                val_f1=float(best_row["val_f1"]),
                save_path=os.path.join(plots_dir, "threshold_optimization.png")
            )
            plot_trend_decay_curves(
                best_test_df,
                threshold=best_th,
                save_path=os.path.join(plots_dir, "trend_decay_curves.png")
            )
            plot_ablation_study(
                save_path=os.path.join(plots_dir, "ablation_study.png")
            )
        except Exception as e:
            logger.warning(f"Could not generate single-model auxiliary plots: {e}")

        # Attach topic name if available
        topic_info_path = self.config.get("topic_info_path", "data/04_topics/topic_info.csv")
        if os.path.exists(topic_info_path):
            try:
                topic_info = pd.read_csv(topic_info_path)
                name_col = "Name" if "Name" in topic_info.columns else ("topic_name" if "topic_name" in topic_info.columns else None)
                id_col = "Topic" if "Topic" in topic_info.columns else ("topic_id" if "topic_id" in topic_info.columns else None)
                if name_col and id_col:
                    name_map = dict(zip(topic_info[id_col], topic_info[name_col]))
                    best_test_df["topic_name"] = best_test_df["topic_id"].map(name_map).fillna("Topic_" + best_test_df["topic_id"].astype(str))
            except Exception as e:
                logger.warning(f"Could not map topic_name: {e}")

        if "topic_name" not in best_test_df.columns and "topic_id" in best_test_df.columns:
            best_test_df["topic_name"] = "Topic_" + best_test_df["topic_id"].astype(str)

        # Save artifacts
        save_dataframe(best_test_df, os.path.join(self.predictions_dir, "all_test_predictions.parquet"))
        trending_topics = best_test_df[best_test_df["predicted_trending"] == 1].sort_values(
            by="trending_probability", ascending=False
        )
        save_dataframe(trending_topics, os.path.join(self.predictions_dir, "predicted_trending_topics.csv"))

        # Save metrics JSON
        metrics_dict = {
            "best_model": best_model_name,
            "models_comparison": comparison_records,
            "forecasting_comparison": ts_comparison
        }
        with open(os.path.join(self.predictions_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4)

        return trending_topics, metrics_dict, comparison_df

    def run(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Main execution entrypoint for Stage 8.
        If compare_models is True, runs the complete 4-model comparison and benchmark suite.
        Otherwise falls back to standard baseline (SARIMA + LightGBM).
        """
        if self.compare_models:
            trending_topics, _, _ = self.run_model_comparison(train_df, valid_df, test_df)
            return trending_topics
        else:
            # Single baseline run (SARIMA + LightGBM)
            logger.info("Executing standard single baseline (SARIMA + LightGBM)...")
            train_fc = self.forecaster_sarima.generate_forecast_features(train_df)
            valid_fc = self.forecaster_sarima.generate_forecast_features(valid_df)
            test_fc = self.forecaster_sarima.generate_forecast_features(test_df)

            X_train, cols = self.fusion.fuse(train_fc, forecaster_type="sarima", is_training=True)
            y_train = train_df["is_trending"].values
            X_valid, _ = self.fusion.fuse(valid_fc, forecaster_type="sarima", is_training=False)
            y_valid = valid_df["is_trending"].values
            X_test, _ = self.fusion.fuse(test_fc, forecaster_type="sarima", is_training=False)
            y_test = test_df["is_trending"].values

            predictor = LightGBMTrendPredictor(self.lgb_config)
            predictor.train(X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names=cols)
            th, val_f1 = self._optimize_threshold(predictor, X_valid, y_valid)
            test_prob = predictor.predict_proba(X_test)
            test_pred = (test_prob >= th).astype(int)

            metrics = evaluate_trend_predictions(y_test, test_pred, test_prob, model_name="SARIMA + LightGBM")
            test_fc["predicted_trending"] = test_pred
            test_fc["trending_probability"] = test_prob
            test_fc = estimate_trend_duration_and_persistence(test_fc, threshold=th)

            save_dataframe(test_fc, os.path.join(self.predictions_dir, "all_test_predictions.parquet"))
            trending = test_fc[test_fc["predicted_trending"] == 1].sort_values(by="trending_probability", ascending=False)
            save_dataframe(trending, os.path.join(self.predictions_dir, "predicted_trending_topics.csv"))
            return trending

    def run_from_splits(self, splits_dir: str = "data/07_splits"):
        train_df = load_dataframe(os.path.join(splits_dir, "train.parquet"))
        valid_df = load_dataframe(os.path.join(splits_dir, "valid.parquet"))
        test_df = load_dataframe(os.path.join(splits_dir, "test.parquet"))
        return self.run(train_df, valid_df, test_df)
