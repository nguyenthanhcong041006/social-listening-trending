import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List
from loguru import logger
from src.utils.file_io import ensure_dir

class LightGBMTrendPredictor:
    """
    Sub-stage 8.3: Trend Prediction (LightGBM)
    Gradient Boosting binary classification model predicting whether a topic will become Trending.
    Input: Combined Feature Vector (Social Listening + SARIMA Forecast Features).
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.params = {
            "objective": self.config.get("objective", "binary"),
            "metric": self.config.get("metric", ["binary_logloss", "auc"]),
            "boosting_type": self.config.get("boosting_type", "gbdt"),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "num_leaves": self.config.get("num_leaves", 31),
            "max_depth": self.config.get("max_depth", -1),
            "feature_fraction": self.config.get("feature_fraction", 0.8),
            "verbose": -1,
            "random_state": 42
        }
        self.n_estimators = self.config.get("n_estimators", 200)
        self.model = None
        self.feature_names = []
        self.evals_result = {}
        self.best_iteration = 0

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        feature_names: List[str] = None
    ):
        """Trains LightGBM model with validation-set early stopping and tracks train/valid/test learning curves."""
        self.feature_names = feature_names or [f"feat_{i}" for i in range(X_train.shape[1])]
        
        # Class imbalance handling per PDF Section 11.3
        params = self.params.copy()
        if "scale_pos_weight" in self.config and self.config["scale_pos_weight"] is not None:
            params["scale_pos_weight"] = float(self.config["scale_pos_weight"])
        elif self.config.get("auto_class_weights", True):
            num_pos = int(np.sum(y_train == 1))
            num_neg = int(np.sum(y_train == 0))
            if num_pos > 0 and num_neg / num_pos > 2.0:
                calc_weight = float(num_neg / num_pos)
                params["scale_pos_weight"] = calc_weight
                logger.info(f"Class imbalance detected (Neg={num_neg}, Pos={num_pos}). Auto-applied scale_pos_weight={calc_weight:.2f}")
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        valid_sets = [train_data]
        valid_names = ["train"]
        
        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, feature_name=self.feature_names)
            valid_sets.append(valid_data)
            valid_names.append("valid")

        if X_test is not None and y_test is not None:
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=self.feature_names)
            valid_sets.append(test_data)
            valid_names.append("test")
            
        self.evals_result = {}
        callbacks = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.record_evaluation(self.evals_result),
            lgb.log_evaluation(period=0)
        ]
        
        logger.info(f"Training LightGBM model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        self.best_iteration = int(getattr(self.model, "best_iteration", 0))
        logger.info(f"LightGBM training completed (Best Iteration: {self.best_iteration}).")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts probability of class Trending = 1."""
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary class: 1 (Trending) or 0 (Not Trending)."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """Extracts feature importance gain from trained LightGBM model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        importances = self.model.feature_importance(importance_type="gain")
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance_gain": importances
        }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)

    def save(self, model_path: str):
        """Saves model weights."""
        ensure_dir(model_path)
        joblib.dump(self.model, model_path)
        logger.info(f"Saved LightGBM model checkpoint to: {model_path}")

    def load(self, model_path: str):
        """Loads model weights."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)
        logger.info(f"Loaded LightGBM model from: {model_path}")
