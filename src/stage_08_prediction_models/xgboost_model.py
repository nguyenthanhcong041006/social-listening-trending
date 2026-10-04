import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, List, Optional
from loguru import logger
from src.utils.file_io import ensure_dir

class XGBoostTrendPredictor:
    """
    Sub-stage 8.3 Alternative: Trend Prediction with eXtreme Gradient Boosting (XGBoost).
    Standard competitive baseline against LightGBM for binary classification of emerging social trends.
    Input: Combined Feature Vector (Social Listening + SARIMA/SARIMAX Forecast Features).
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.n_estimators = self.config.get("n_estimators", 200)
        self.learning_rate = self.config.get("learning_rate", 0.05)
        self.max_depth = self.config.get("max_depth", 5)
        self.subsample = self.config.get("subsample", 0.8)
        self.colsample_bytree = self.config.get("colsample_bytree", 0.8)
        self.early_stopping_rounds = self.config.get("early_stopping_rounds", 20)
        self.random_state = self.config.get("random_state", 42)

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.evals_result: Dict[str, Any] = {}
        self.best_iteration: int = 0

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Trains XGBoost model with validation early stopping and records train/valid/test learning trajectories.
        """
        self.feature_names = feature_names or [f"feat_{i}" for i in range(X_train.shape[1])]

        # Automatic class imbalance handling
        scale_pos_weight = 1.0
        if "scale_pos_weight" in self.config and self.config["scale_pos_weight"] is not None:
            scale_pos_weight = float(self.config["scale_pos_weight"])
        elif self.config.get("auto_class_weights", True):
            num_pos = int(np.sum(y_train == 1))
            num_neg = int(np.sum(y_train == 0))
            if num_pos > 0 and num_neg / num_pos > 1.5:
                scale_pos_weight = float(num_neg / num_pos)
                logger.info(f"[XGBoost] Imbalance detected (Neg={num_neg}, Pos={num_pos}). Scale_pos_weight={scale_pos_weight:.2f}")

        eval_set = [(X_train, y_train)]
        eval_names = ["train"]

        if X_valid is not None and y_valid is not None:
            eval_set.append((X_valid, y_valid))
            eval_names.append("valid")

        if X_test is not None and y_test is not None:
            eval_set.append((X_test, y_test))
            eval_names.append("test")

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            eval_metric=["logloss", "auc"],
            early_stopping_rounds=self.early_stopping_rounds if X_valid is not None else None,
            random_state=self.random_state,
            n_jobs=-1
        )

        logger.info(f"Training XGBoost model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )

        self.best_iteration = getattr(self.model, "best_iteration", self.n_estimators) or self.n_estimators
        
        # Standardize evals_result format to match LightGBM {split_name: {metric_name: list}}
        raw_evals = self.model.evals_result()
        formatted_evals = {}
        for idx, split_key in enumerate(raw_evals.keys()):
            name = eval_names[idx] if idx < len(eval_names) else split_key
            split_metrics = {}
            for metric_k, val_list in raw_evals[split_key].items():
                std_key = "binary_logloss" if metric_k == "logloss" else metric_k
                split_metrics[std_key] = val_list
            formatted_evals[name] = split_metrics

        self.evals_result = formatted_evals
        logger.info(f"XGBoost training completed (Best Iteration: {self.best_iteration}).")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts probability of class Trending = 1."""
        if self.model is None:
            raise ValueError("XGBoost model has not been trained yet!")
        probs = self.model.predict_proba(X)
        return probs[:, 1] if probs.ndim == 2 else probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary class: 1 (Trending) or 0 (Not Trending)."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """Extracts feature importance gain from trained XGBoost model."""
        if self.model is None:
            raise ValueError("XGBoost model has not been trained yet!")
        
        booster = self.model.get_booster()
        score_gain = booster.get_score(importance_type="gain")
        
        importances = []
        for i, f_name in enumerate(self.feature_names):
            key = f"f{i}"
            val = score_gain.get(key, score_gain.get(f_name, 0.0))
            importances.append(val)

        return pd.DataFrame({
            "feature": self.feature_names,
            "importance_gain": importances
        }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)

    def save(self, model_path: str):
        """Saves model checkpoint."""
        ensure_dir(model_path)
        joblib.dump(self.model, model_path)
        logger.info(f"Saved XGBoost model checkpoint to: {model_path}")

    def load(self, model_path: str):
        """Loads model checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)
        logger.info(f"Loaded XGBoost model from: {model_path}")
