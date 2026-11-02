import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from src.utils.file_io import ensure_dir

class RandomForestTrendPredictor:
    """
    Stage 8 Alternative / Competitive ML Baseline: Trend Prediction with Random Forest.
    A Bagging Ensemble of decision trees trained with bootstrap subsampling and feature random subspaces.
    Replaces heuristic persistence rules with a robust, non-linear machine learning classifier that avoids
    gradient boosting over-sensitivity on moderate-sized multi-platform feature spaces.
    Input: Combined Multi-Platform Feature Vector (Social Listening + SARIMA/SARIMAX Forecast Features).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.n_estimators = int(self.config.get("n_estimators", 150))
        self.max_depth = self.config.get("max_depth", 5)
        if self.max_depth is not None:
            self.max_depth = int(self.max_depth)
        self.min_samples_split = int(self.config.get("min_samples_split", 5))
        self.min_samples_leaf = int(self.config.get("min_samples_leaf", 2))
        self.max_features = self.config.get("max_features", "sqrt")
        self.class_weight = self.config.get("class_weight", "balanced_subsample")
        self.bootstrap = bool(self.config.get("bootstrap", True))
        self.oob_score = bool(self.config.get("oob_score", True))
        self.random_state = int(self.config.get("random_state", 42))
        self.n_jobs = int(self.config.get("n_jobs", -1))

        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = []
        self.evals_result: Dict[str, Any] = {}
        self.best_iteration: int = self.n_estimators

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
        Trains RandomForestClassifier and tracks ensemble convergence curves across tree additions.
        """
        self.feature_names = feature_names or [f"feat_{i}" for i in range(X_train.shape[1])]

        num_pos = int(np.sum(y_train == 1))
        num_neg = int(np.sum(y_train == 0))
        logger.info(
            f"[RandomForest] Initializing Bagging Ensemble with {self.n_estimators} trees "
            f"(max_depth={self.max_depth}, max_features={self.max_features}, class_weight={self.class_weight}). "
            f"Train samples: {X_train.shape[0]} (Neg={num_neg}, Pos={num_pos})."
        )

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        logger.info(f"Training Random Forest model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self.model.fit(X_train, y_train)

        if self.oob_score and hasattr(self.model, "oob_score_"):
            logger.info(f"[RandomForest] Out-Of-Bag (OOB) Accuracy: {self.model.oob_score_:.4f}")

        # Compute ensemble convergence trajectory for learning curves visualization
        eval_splits = [("train", X_train, y_train)]
        if X_valid is not None and y_valid is not None:
            eval_splits.append(("valid", X_valid, y_valid))
        if X_test is not None and y_test is not None:
            eval_splits.append(("test", X_test, y_test))

        self.evals_result = {name: {"binary_logloss": [], "auc": []} for name, _, _ in eval_splits}
        
        # Track predictions across trees efficiently
        try:
            # Sample evaluation steps (e.g. every 5 trees and the last tree)
            eval_steps = sorted(list(set(list(range(5, self.n_estimators + 1, 5)) + [self.n_estimators])))
            
            # Precompute tree probability matrices for each split
            for name, X_split, y_split in eval_splits:
                # Accumulate tree predictions
                running_prob_sum = np.zeros(len(y_split), dtype=float)
                prev_step = 0
                for step in eval_steps:
                    for t_idx in range(prev_step, step):
                        tree = self.model.estimators_[t_idx]
                        tree_probs = tree.predict_proba(X_split)
                        if tree_probs.shape[1] == 2:
                            running_prob_sum += tree_probs[:, 1]
                        else:
                            # Single class edge case in individual tree
                            tree_class = tree.classes_[0]
                            running_prob_sum += (tree_class == 1).astype(float)
                    prev_step = step

                    avg_prob = np.clip(running_prob_sum / step, 1e-7, 1.0 - 1e-7)
                    
                    try:
                        loss = float(log_loss(y_split, avg_prob))
                    except Exception:
                        loss = 0.5
                    
                    try:
                        if len(np.unique(y_split)) > 1:
                            auc_val = float(roc_auc_score(y_split, avg_prob))
                        else:
                            auc_val = 0.5
                    except Exception:
                        auc_val = 0.5

                    self.evals_result[name]["binary_logloss"].append(loss)
                    self.evals_result[name]["auc"].append(auc_val)
        except Exception as e:
            logger.warning(f"[RandomForest] Could not compute tree trajectory for learning curves: {e}")
            # Fallback with final point
            for name, X_split, y_split in eval_splits:
                p = self.predict_proba(X_split)
                loss = float(log_loss(y_split, np.clip(p, 1e-7, 1.0 - 1e-7)))
                auc_val = float(roc_auc_score(y_split, p)) if len(np.unique(y_split)) > 1 else 0.5
                self.evals_result[name]["binary_logloss"] = [loss]
                self.evals_result[name]["auc"] = [auc_val]

        self.best_iteration = self.n_estimators
        logger.info(f"Random Forest training completed ({self.n_estimators} trees fitted).")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts probability of class Trending = 1."""
        if self.model is None:
            raise ValueError("Random Forest model has not been trained yet!")
        probs = self.model.predict_proba(X)
        if probs.ndim == 2:
            if probs.shape[1] == 2:
                return probs[:, 1]
            elif probs.shape[1] == 1:
                return np.ones(len(X)) if self.model.classes_[0] == 1 else np.zeros(len(X))
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary class: 1 (Trending) or 0 (Not Trending)."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """Extracts Mean Decrease in Impurity (Gini importance) from trained Random Forest."""
        if self.model is None:
            raise ValueError("Random Forest model has not been trained yet!")

        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance_gain": importances
        }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)

    def save(self, model_path: str):
        """Saves model checkpoint."""
        ensure_dir(model_path)
        joblib.dump(self.model, model_path)
        logger.info(f"Saved Random Forest model checkpoint to: {model_path}")

    def load(self, model_path: str):
        """Loads model checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)
        logger.info(f"Loaded Random Forest model from: {model_path}")
