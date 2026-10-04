import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Tuple, Optional
from loguru import logger

class FeatureFusion:
    """
    Sub-stage 8.2: Feature Fusion
    Fuses two independent feature streams into a unified representation:
    1. Social Listening Features:
       - topic_volume, growth_rate, engagement, avg_sentiment, sentiment_change,
         positive_ratio, negative_ratio, neutral_ratio, hashtag_activity, hashtag_growth,
         acceleration, unique_users
       - Lag & rolling statistics
    2. Time-Series Forecast Features:
       - SARIMA: sarima_forecast_volume, sarima_forecast_growth
       - SARIMAX: sarimax_forecast_volume, sarimax_forecast_growth
    
    Output: Combined Feature Vector fed directly into the gradient boosting classifiers (LightGBM / XGBoost).
    """

    SOCIAL_LISTENING_FEATURES = [
        "topic_volume",
        "growth_rate",
        "engagement",
        "avg_sentiment",
        "sentiment_change",
        "positive_ratio",
        "negative_ratio",
        "neutral_ratio",
        "hashtag_activity",
        "hashtag_growth",
        "acceleration",
        "unique_users",
        "volume_lag_1",
        "growth_rate_lag_1",
        "engagement_lag_1",
        "rolling_mean_volume_3",
        "rolling_mean_engagement_3",
    ]

    SARIMA_FEATURES = [
        "sarima_forecast_volume",
        "sarima_forecast_growth",
    ]

    SARIMAX_FEATURES = [
        "sarimax_forecast_volume",
        "sarimax_forecast_growth",
    ]

    def __init__(self, scale_features: bool = False, scaler_type: str = "standard"):
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self.fitted_scaler = False

    def fuse(
        self,
        df: pd.DataFrame,
        forecaster_type: str = "auto",
        is_training: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extracts and concatenates feature columns into the Combined Feature Vector.
        forecaster_type: "sarima", "sarimax", "both", or "auto" (detects available in df).
        """
        available_social = [col for col in self.SOCIAL_LISTENING_FEATURES if col in df.columns]

        if forecaster_type == "sarima":
            ts_cols = [col for col in self.SARIMA_FEATURES if col in df.columns]
        elif forecaster_type == "sarimax":
            ts_cols = [col for col in self.SARIMAX_FEATURES if col in df.columns]
        elif forecaster_type == "both":
            ts_cols = [col for col in (self.SARIMA_FEATURES + self.SARIMAX_FEATURES) if col in df.columns]
        else: # "auto"
            ts_cols = [col for col in (self.SARIMA_FEATURES + self.SARIMAX_FEATURES) if col in df.columns]

        feature_cols = available_social + ts_cols
        X = df[feature_cols].fillna(0.0).values

        if self.scale_features:
            if is_training or not self.fitted_scaler:
                X = self.scaler.fit_transform(X)
                self.fitted_scaler = True
            else:
                X = self.scaler.transform(X)

        logger.debug(f"Fused {X.shape[1]} features ({len(available_social)} Social + {len(ts_cols)} Forecast: {forecaster_type}).")
        return X, feature_cols
