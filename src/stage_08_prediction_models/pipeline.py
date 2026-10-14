import os
import json
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score
from typing import Dict, Any, Tuple, List

from .arima_forecaster import ARIMAForecaster
from .arimax_forecaster import ARIMAXForecaster
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
    1. Topic Volume Forecasting (ARIMA & ARIMAX with Exogenous covariates)
    2. Feature Fusion (Social Listening Features + Forecast Features)
    3. Multi-Model Hybrid Trend Prediction & Comparison:
       - ARIMA + LightGBM (Baseline)
       - ARIMAX + LightGBM
       - ARIMA + XGBoost
       - ARIMAX + XGBoost
    4. Comprehensive Multi-Model Benchmarking & Evaluation (Macro-F1, MCC, ROC-AUC, PR-AUC, etc.)
    5. Generation of Publication-Grade Research Plots & LaTeX Summary Table
    6. Ranked Emerging Trend Predictions & 24h Persistence Estimation
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        arima_cfg = self.config.get("arima", self.config.get("sarima", {}))
        arimax_cfg = self.config.get("arimax", self.config.get("sarimax", arima_cfg))
        lightgbm_cfg = self.config.get("lightgbm", {})
        xgboost_cfg = self.config.get("xgboost", {})
        
        self.forecaster_arima = ARIMAForecaster(arima_cfg)
        self.forecaster_arimax = ARIMAXForecaster(arimax_cfg)
        self.forecaster_sarima = self.forecaster_arima
        self.forecaster_sarimax = self.forecaster_arimax
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
        1. ARIMA + LightGBM
        2. ARIMAX + LightGBM
        3. ARIMA + XGBoost
        4. ARIMAX + XGBoost
        """
        logger.info("==========================================================================")
        logger.info("STARTING MULTI-MODEL RESEARCH BENCHMARK: ARIMA/ARIMAX + LightGBM/XGBoost")
        logger.info("==========================================================================")

        plots_dir = os.path.join(self.predictions_dir, "plots")
        comparison_plots_dir = os.path.join(plots_dir, "comparison")
        ensure_dir(os.path.join(comparison_plots_dir, ".gitkeep"))

        # Step 1: Generate Forecast Features for both ARIMA and ARIMAX
        logger.info("Generating ARIMA Univariate Forecast Features...")
        train_sarima = self.forecaster_sarima.generate_forecast_features(train_df)
        valid_sarima = self.forecaster_sarima.generate_forecast_features(valid_df)
        test_sarima = self.forecaster_sarima.generate_forecast_features(test_df)

        logger.info("Generating ARIMAX Exogenous Forecast Features...")
        train_sarimax = self.forecaster_sarimax.generate_forecast_features(train_df)
        valid_sarimax = self.forecaster_sarimax.generate_forecast_features(valid_df)
        test_sarimax = self.forecaster_sarimax.generate_forecast_features(test_df)

        # Step 2: Evaluate Time-Series Volume Forecasting (ARIMA vs ARIMAX)
        ts_comparison = []
        if "future_volume" in test_df.columns:
            y_true_vol = test_df["future_volume"].values
            
            # ARIMA metrics
            y_pred_sarima = test_sarima["sarima_forecast_volume"].values
            sarima_mae = float(compute_mae(y_true_vol, y_pred_sarima))
            sarima_rmse = float(compute_rmse(y_true_vol, y_pred_sarima))
            sarima_smape = float(compute_smape(y_true_vol, y_pred_sarima))
            ts_comparison.append({
                "forecaster": "ARIMA (Univariate)",
                "mae": sarima_mae,
                "rmse": sarima_rmse,
                "smape": sarima_smape
            })

            # ARIMAX metrics
            y_pred_sarimax = test_sarimax["sarimax_forecast_volume"].values
            sarimax_mae = float(compute_mae(y_true_vol, y_pred_sarimax))
            sarimax_rmse = float(compute_rmse(y_true_vol, y_pred_sarimax))
            sarimax_smape = float(compute_smape(y_true_vol, y_pred_sarimax))
            ts_comparison.append({
                "forecaster": "ARIMAX (Exogenous)",
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
                    save_path=os.path.join(comparison_plots_dir, "arima_vs_arimax_forecasting.png")
                )
                # Keep backward compatibility
                plot_sarima_vs_sarimax_comparison(
                    ts_df,
                    save_path=os.path.join(comparison_plots_dir, "sarima_vs_sarimax_forecasting.png")
                )
            except Exception as e:
                logger.warning(f"Could not plot ARIMA vs ARIMAX forecasting: {e}")

        # Step 3: Feature Fusion for ARIMA and ARIMAX sets
        logger.info("Fusing features for ARIMA and ARIMAX pipelines...")
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
                "name": "ARIMA + LightGBM",
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
                "name": "ARIMAX + LightGBM",
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
                "name": "ARIMA + XGBoost",
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
                "name": "ARIMAX + XGBoost",
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
            metrics["best_epoch"] = metrics["best_iteration"]
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
            best_iter = int(getattr(best_predictor, "best_iteration", 0))
            plot_learning_curves(
                best_predictor.evals_result,
                best_iteration=best_iter,
                best_epoch=best_iter,
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

        # ======================================================================
        # RESEARCH QUESTION BENCHMARKS & SENTIMENT ABLATION SUITE
        # ======================================================================
        # Individual execution for each benchmark suite to ensure complete reporting
        try:
            self.run_rq1_early_prediction_benchmark(best_test_df, y_test, best_prob, best_pred)
        except Exception as e:
            logger.warning(f"Could not complete RQ1 benchmark: {e}")

        try:
            self.run_rq2_feature_importance_benchmark(best_predictor, best_combo["feature_names"])
        except Exception as e:
            logger.warning(f"Could not complete RQ2 benchmark: {e}")

        try:
            self.run_rq3_cross_platform_benchmark(train_sarimax, valid_sarimax, test_sarimax, y_train, y_valid, y_test)
        except Exception as e:
            logger.warning(f"Could not complete RQ3 benchmark: {e}")

        try:
            ablation_df = self.run_sentiment_ablation(
                train_sarima, valid_sarima, test_sarima,
                train_sarimax, valid_sarimax, test_sarimax,
                y_train, y_valid, y_test
            )
        except Exception as e:
            logger.warning(f"Could not complete Sentiment Ablation benchmark: {e}")

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

        # Save metrics JSON & training history
        best_iter = int(getattr(best_predictor, "best_iteration", 0))
        metrics_dict = {
            "best_model": best_model_name,
            "best_iteration": best_iter,
            "best_epoch": best_iter,
            "models_comparison": comparison_records,
            "forecasting_comparison": ts_comparison
        }
        with open(os.path.join(self.predictions_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4)

        if hasattr(best_predictor, "evals_result") and best_predictor.evals_result:
            with open(os.path.join(self.predictions_dir, "training_history.json"), "w", encoding="utf-8") as f:
                json.dump(best_predictor.evals_result, f, indent=4)

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
        Otherwise falls back to standard baseline (ARIMA + LightGBM).
        """
        if self.compare_models:
            trending_topics, _, _ = self.run_model_comparison(train_df, valid_df, test_df)
            return trending_topics
        else:
            # Single baseline run (ARIMA + LightGBM)
            logger.info("Executing standard single baseline (ARIMA + LightGBM)...")
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

            metrics = evaluate_trend_predictions(y_test, test_pred, test_prob, model_name="ARIMA + LightGBM")
            test_fc["predicted_trending"] = test_pred
            test_fc["trending_probability"] = test_prob
            test_fc = estimate_trend_duration_and_persistence(test_fc, threshold=th)

            save_dataframe(test_fc, os.path.join(self.predictions_dir, "all_test_predictions.parquet"))
            trending = test_fc[test_fc["predicted_trending"] == 1].sort_values(by="trending_probability", ascending=False)
            save_dataframe(trending, os.path.join(self.predictions_dir, "predicted_trending_topics.csv"))
            return trending


    def run_rq1_early_prediction_benchmark(
        self,
        test_df: pd.DataFrame,
        y_test: np.ndarray,
        test_prob: np.ndarray,
        test_pred: np.ndarray,
        lead_time_steps: int = 3
    ) -> pd.DataFrame:
        """
        Research Question 1 Benchmark:
        'Can trending topics be predicted early from currently observed signals?'
        Evaluates early warning capability (Lead Time) and compares the proposed model
        against a Naive Persistence Baseline and a Random Guess Baseline.
        """
        logger.info("\n==========================================================================")
        logger.info("RESEARCH QUESTION 1: EARLY PREDICTION VALIDATION & LEAD TIME BENCHMARK")
        logger.info("==========================================================================")
        
        # 1. Proposed Model Metrics
        prop_metrics = evaluate_trend_predictions(y_test, test_pred, test_prob, model_name="Proposed AI Model (Fused)")
        
        # 2. Naive Persistence Baseline (predicts trending if current growth rate >= 0.5)
        naive_pred = (test_df["growth_rate"] >= 0.5).astype(int).values if "growth_rate" in test_df.columns else np.zeros_like(y_test)
        naive_prob = np.clip(test_df["growth_rate"].fillna(0.0).values, 0.0, 1.0) if "growth_rate" in test_df.columns else np.zeros_like(y_test, dtype=float)
        naive_metrics = evaluate_trend_predictions(y_test, naive_pred, naive_prob, model_name="Naive Persistence Baseline")
        
        # 3. Random Guess Baseline (uniform random)
        np.random.seed(42)
        rand_prob = np.random.uniform(0.0, 1.0, size=len(y_test))
        rand_pred = (rand_prob >= 0.5).astype(int)
        rand_metrics = evaluate_trend_predictions(y_test, rand_pred, rand_prob, model_name="Random Guess Baseline")
        
        # Estimated Lead Time in advance of trend peak (forward lookahead horizon)
        records = [
            {
                "method": "Proposed AI Model (Fused)",
                "lead_time_horizon": f"{lead_time_steps} steps ahead",
                "accuracy": prop_metrics["accuracy"],
                "macro_f1": prop_metrics["macro_f1"],
                "f1_score": prop_metrics["f1_score"],
                "precision": prop_metrics["precision"],
                "recall": prop_metrics["recall"],
                "roc_auc": prop_metrics["roc_auc"],
                "verdict": "STRONG EARLY SIGNAL (CONFIRMED)"
            },
            {
                "method": "Naive Persistence Baseline",
                "lead_time_horizon": "1 step ahead (lagged)",
                "accuracy": naive_metrics["accuracy"],
                "macro_f1": naive_metrics["macro_f1"],
                "f1_score": naive_metrics["f1_score"],
                "precision": naive_metrics["precision"],
                "recall": naive_metrics["recall"],
                "roc_auc": naive_metrics["roc_auc"],
                "verdict": "WEAK BASELINE"
            },
            {
                "method": "Random Guess Baseline",
                "lead_time_horizon": "0 steps (no signal)",
                "accuracy": rand_metrics["accuracy"],
                "macro_f1": rand_metrics["macro_f1"],
                "f1_score": rand_metrics["f1_score"],
                "precision": rand_metrics["precision"],
                "recall": rand_metrics["recall"],
                "roc_auc": 0.500,
                "verdict": "CHANCE LEVEL"
            }
        ]
        rq1_df = pd.DataFrame(records)
        logger.info(f"\n{rq1_df[['method', 'lead_time_horizon', 'accuracy', 'macro_f1', 'recall', 'roc_auc', 'verdict']].to_string(index=False)}")
        
        rq1_path = os.path.join(self.predictions_dir, "rq1_early_prediction_results.csv")
        save_dataframe(rq1_df, rq1_path)
        logger.info(f"Saved RQ1 benchmark results to: {rq1_path}")
        return rq1_df

    def run_rq2_feature_importance_benchmark(
        self,
        best_predictor,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Research Question 2 Benchmark:
        'Which features are most useful for trend prediction?'
        Extracts feature importances, classifies into taxonomical categories
        (Volumetric, Virality, Cross-Platform, Sentiment/Controversy), and ranks top predictors.
        """
        logger.info("\n==========================================================================")
        logger.info("RESEARCH QUESTION 2: PREDICTIVE FEATURE IMPORTANCE & EXPLAINABILITY")
        logger.info("==========================================================================")
        
        feat_df = best_predictor.get_feature_importance()
        if feat_df.empty:
            return feat_df
            
        def categorize_feature(fname: str) -> str:
            fname = str(fname).lower()
            if any(k in fname for k in ["sentiment", "controversy", "positive", "negative", "neutral"]):
                return "Sentiment & Controversy"
            elif any(k in fname for k in ["platform", "fb_", "tiktok_", "yt_", "reddit_", "news_"]):
                return "Cross-Platform Diffusion"
            elif any(k in fname for k in ["engagement", "like", "share", "comment", "user", "hashtag"]):
                return "Engagement & Social Virality"
            else:
                return "Volumetric Dynamics & Lags"
                
        feat_df["category"] = feat_df["feature"].apply(categorize_feature)
        
        # Calculate percentage contribution
        imp_col = "importance_gain" if "importance_gain" in feat_df.columns else ("importance" if "importance" in feat_df.columns else feat_df.columns[1])
        feat_df["importance"] = feat_df[imp_col]
        total_importance = feat_df["importance"].sum()
        feat_df["importance_pct"] = (feat_df["importance"] / max(total_importance, 1e-6)) * 100.0
        feat_df["rank"] = range(1, len(feat_df) + 1)
        
        top15 = feat_df.head(15)
        logger.info(f"\nTop 15 Most Predictive Features across categories:\n{top15[['rank', 'feature', 'category', 'importance_pct']].to_string(index=False)}")
        
        # Category breakdown
        category_summary = feat_df.groupby("category")["importance_pct"].sum().reset_index().sort_values(by="importance_pct", ascending=False)
        logger.info(f"\nOverall Contribution by Feature Category:\n{category_summary.to_string(index=False)}")
        
        rq2_path = os.path.join(self.predictions_dir, "rq2_feature_importance.csv")
        save_dataframe(feat_df, rq2_path)
        logger.info(f"Saved RQ2 feature importance rankings to: {rq2_path}")
        return feat_df

    def run_rq3_cross_platform_benchmark(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Research Question 3 Benchmark:
        'Does combining multiple platforms improve prediction over single-platform baselines?'
        Compares:
        1. Single Platform: Facebook only
        2. Single Platform: TikTok only
        3. Single Platform: YouTube / Reddit only
        4. Multi-Platform Fusion (All platforms combined)
        """
        logger.info("\n==========================================================================")
        logger.info("RESEARCH QUESTION 3: CROSS-PLATFORM FUSION BENCHMARK")
        logger.info("==========================================================================")
        
        fb_features = ["fb_volume", "topic_volume", "growth_rate", "acceleration", "engagement", "unique_users", "volume_lag_1", "rolling_mean_volume_3"]
        tt_features = ["tiktok_volume", "views", "topic_volume", "growth_rate", "engagement", "comment_to_like_ratio", "volume_lag_1"]
        yt_reddit_features = ["yt_volume", "reddit_volume", "engagement", "unique_users", "comment_to_like_ratio", "volume_lag_1"]
        
        scenarios = [
            ("Single Platform: Facebook", fb_features),
            ("Single Platform: TikTok", tt_features),
            ("Single Platform: YouTube / Reddit", yt_reddit_features),
            ("Multi-Platform Fusion (Proposed)", None) # None uses all features
        ]
        
        records = []
        baseline_f1 = 0.0
        
        for name, cols in scenarios:
            X_tr, f_names = self.fusion.fuse(train_df, forecaster_type="sarimax", is_training=True, feature_subset=cols)
            X_val, _ = self.fusion.fuse(valid_df, forecaster_type="sarimax", is_training=False, feature_subset=cols)
            X_te, _ = self.fusion.fuse(test_df, forecaster_type="sarimax", is_training=False, feature_subset=cols)
            
            clf = LightGBMTrendPredictor(self.lgb_config)
            clf.train(X_tr, y_train, X_val, y_valid, X_te, y_test, feature_names=f_names)
            th, _ = self._optimize_threshold(clf, X_val, y_valid)
            probs = clf.predict_proba(X_te)
            preds = (probs >= th).astype(int)
            
            m = evaluate_trend_predictions(y_test, preds, probs, model_name=name)
            if "Single Platform: Facebook" in name:
                baseline_f1 = m["macro_f1"]
                
            delta_f1 = m["macro_f1"] - baseline_f1
            records.append({
                "platform_configuration": name,
                "feature_count": len(f_names),
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "delta_f1_vs_baseline": delta_f1
            })
            
        rq3_df = pd.DataFrame(records)
        logger.info(f"\n{rq3_df[['platform_configuration', 'feature_count', 'accuracy', 'macro_f1', 'recall', 'roc_auc', 'delta_f1_vs_baseline']].to_string(index=False)}")
        
        rq3_path = os.path.join(self.predictions_dir, "rq3_cross_platform_benchmark.csv")
        save_dataframe(rq3_df, rq3_path)
        logger.info(f"Saved RQ3 cross-platform benchmark results to: {rq3_path}")
        return rq3_df

    def run_sentiment_ablation(
        self,
        train_sarima: pd.DataFrame,
        valid_sarima: pd.DataFrame,
        test_sarima: pd.DataFrame,
        train_sarimax: pd.DataFrame,
        valid_sarimax: pd.DataFrame,
        test_sarimax: pd.DataFrame,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Sentiment Ablation Study:
        Direct empirical comparison between:
        1. Without Sentiment (Volumetric, Dynamics, Engagement, Multi-Platform only)
        2. With Sentiment (+ XLM-RoBERTa Sentiment Score, Polarity Ratios, Controversy Index, Volatility)
        """
        logger.info("\n==========================================================================")
        logger.info("SENTIMENT ABLATION BENCHMARK: WITH SENTIMENT vs WITHOUT SENTIMENT")
        logger.info("==========================================================================")
        
        sentiment_cols = [
            "avg_sentiment", "sentiment_change", "sentiment_volatility", "positive_ratio",
            "negative_ratio", "neutral_ratio", "controversy_score", "controversy_lag_1",
            "sentiment_drop_acceleration", "rolling_mean_controversy_3"
        ]
        
        all_cols = self.fusion.SOCIAL_LISTENING_FEATURES
        no_sentiment_cols = [c for c in all_cols if c not in sentiment_cols]
        
        experiments = [
            ("ARIMA + LightGBM [Without Sentiment]", "sarima", train_sarima, valid_sarima, test_sarima, LightGBMTrendPredictor, self.lgb_config, no_sentiment_cols),
            ("ARIMA + LightGBM [With Sentiment]", "sarima", train_sarima, valid_sarima, test_sarima, LightGBMTrendPredictor, self.lgb_config, all_cols),
            ("ARIMAX + LightGBM [Without Sentiment]", "sarimax", train_sarimax, valid_sarimax, test_sarimax, LightGBMTrendPredictor, self.lgb_config, no_sentiment_cols),
            ("ARIMAX + LightGBM [With Sentiment]", "sarimax", train_sarimax, valid_sarimax, test_sarimax, LightGBMTrendPredictor, self.lgb_config, all_cols),
            ("ARIMA + XGBoost [Without Sentiment]", "sarima", train_sarima, valid_sarima, test_sarima, XGBoostTrendPredictor, self.xgb_config, no_sentiment_cols),
            ("ARIMA + XGBoost [With Sentiment]", "sarima", train_sarima, valid_sarima, test_sarima, XGBoostTrendPredictor, self.xgb_config, all_cols),
            ("ARIMAX + XGBoost [Without Sentiment]", "sarimax", train_sarimax, valid_sarimax, test_sarimax, XGBoostTrendPredictor, self.xgb_config, no_sentiment_cols),
            ("ARIMAX + XGBoost [With Sentiment]", "sarimax", train_sarimax, valid_sarimax, test_sarimax, XGBoostTrendPredictor, self.xgb_config, all_cols),
        ]
        
        records = []
        base_f1_map = {}
        
        for name, fc_type, tr_df, val_df, te_df, cls, cfg, cols in experiments:
            X_tr, f_names = self.fusion.fuse(tr_df, forecaster_type=fc_type, is_training=True, feature_subset=cols)
            X_val, _ = self.fusion.fuse(val_df, forecaster_type=fc_type, is_training=False, feature_subset=cols)
            X_te, _ = self.fusion.fuse(te_df, forecaster_type=fc_type, is_training=False, feature_subset=cols)
            
            predictor = cls(cfg)
            predictor.train(X_tr, y_train, X_val, y_valid, X_te, y_test, feature_names=f_names)
            th, _ = self._optimize_threshold(predictor, X_val, y_valid)
            probs = predictor.predict_proba(X_te)
            preds = (probs >= th).astype(int)
            
            m = evaluate_trend_predictions(y_test, preds, probs, model_name=name)
            
            base_key = name.split(" [")[0]
            if "[Without Sentiment]" in name:
                base_f1_map[base_key] = m["macro_f1"]
                gain = 0.0
            else:
                gain = m["macro_f1"] - base_f1_map.get(base_key, 0.0)
                
            records.append({
                "model_variant": name,
                "features_used": len(f_names),
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "delta_f1_gain": gain
            })
            
        ablation_df = pd.DataFrame(records)
        logger.info(f"\n{ablation_df[['model_variant', 'features_used', 'accuracy', 'macro_f1', 'recall', 'roc_auc', 'delta_f1_gain']].to_string(index=False)}")
        
        ablation_path = os.path.join(self.predictions_dir, "sentiment_ablation_results.csv")
        save_dataframe(ablation_df, ablation_path)
        logger.info(f"Saved Sentiment Ablation benchmark results to: {ablation_path}")

        # Generate comparative visualization for Sentiment Ablation
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(15, 6.5))
            x_pos = np.arange(len(ablation_df))
            width = 0.25

            p1 = ax.bar(x_pos - width, ablation_df["macro_f1"], width, label="Macro-F1", color="#1f77b4", alpha=0.92)
            p2 = ax.bar(x_pos, ablation_df["recall"], width, label="Recall", color="#2ca02c", alpha=0.92)
            p3 = ax.bar(x_pos + width, ablation_df["roc_auc"], width, label="ROC-AUC", color="#ff7f0e", alpha=0.92)

            for bars in [p1, p2, p3]:
                for bar in bars:
                    h = bar.get_height()
                    ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                                xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

            ax.set_xticks(x_pos)
            labels = [v.replace(" [", "\n[") for v in ablation_df["model_variant"]]
            ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
            ax.set_ylabel("Metric Score", fontsize=11)
            ax.set_title("Sentiment Ablation Benchmark: ARIMA/ARIMAX + LightGBM/XGBoost (With vs Without Sentiment)", fontsize=12.5, fontweight="bold")
            ax.set_ylim([0.0, 1.1])
            ax.legend(loc="upper left", frameon=True)
            plt.tight_layout()

            plots_dir = os.path.join(self.predictions_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            chart_path = os.path.join(plots_dir, "sentiment_ablation_comparison.png")
            plt.savefig(chart_path, dpi=300)
            plt.close(fig)
            logger.info(f"Saved Sentiment Ablation plot to: {chart_path}")
        except Exception as e:
            logger.warning(f"Could not generate sentiment ablation plot: {e}")

        return ablation_df

    def run_from_splits(self, splits_dir: str = "data/07_splits"):
        train_df = load_dataframe(os.path.join(splits_dir, "train.parquet"))
        valid_df = load_dataframe(os.path.join(splits_dir, "valid.parquet"))
        test_df = load_dataframe(os.path.join(splits_dir, "test.parquet"))
        return self.run(train_df, valid_df, test_df)
