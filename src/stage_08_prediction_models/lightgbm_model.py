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
            "metric": self.config.get("metric", "auc"),
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

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
        feature_names: List[str] = None
    ):
        """Trains LightGBM model with validation-set early stopping."""
        self.feature_names = feature_names or [f"feat_{i}" for i in range(X_train.shape[1])]
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        valid_sets = [train_data]
        valid_names = ["train"]
        
        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, feature_name=self.feature_names)
            valid_sets.append(valid_data)
            valid_names.append("valid")
            
        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False), lgb.log_evaluation(period=0)]
        
        logger.info(f"Training LightGBM model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        logger.info("LightGBM model training completed.")

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
