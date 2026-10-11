import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Tuple, Optional
from loguru import logger

class FeatureFusion:
    """
    Sub-stage 8.2: Feature Fusion for Multi-Platform Sports & Early Trend Prediction
    Fuses two independent feature streams into a unified representation:
    1. Social Listening Features:
       - Volumetric: topic_volume, growth_rate, acceleration, lags 1-3, rolling_mean_volume_3
       - Social Virality: engagement, comment_to_like_ratio, unique_users, rolling_mean_engagement_3
       - Multi-Platform Diffusion: platform_count, cross_platform_entropy, platform_spread_speed,
         fb_volume, tiktok_volume, yt_volume, reddit_volume, news_volume
       - Sentiment & Controversy (Sports/Drama): avg_sentiment, sentiment_change, sentiment_volatility,
         positive_ratio, negative_ratio, neutral_ratio, controversy_score, controversy_lag_1,
         sentiment_drop_acceleration, rolling_mean_controversy_3
    2. Time-Series Forecast Features:
       - SARIMA: sarima_forecast_volume, sarima_forecast_growth
       - SARIMAX: sarimax_forecast_volume, sarimax_forecast_growth
    """

    SOCIAL_LISTENING_FEATURES = [
        "topic_volume",
        "growth_rate",
        "acceleration",
        "engagement",
        "views",
        "comment_to_like_ratio",
        "unique_users",
        "hashtag_activity",
        "hashtag_growth",
        # Multi-Platform Diffusion
        "platform_count",
        "cross_platform_entropy",
        "platform_spread_speed",
        "fb_volume",
        "tiktok_volume",
        "yt_volume",
        "reddit_volume",
        "news_volume",
        # Sentiment & Sports Controversy Signals
        "avg_sentiment",
        "sentiment_change",
        "sentiment_volatility",
        "positive_ratio",
        "negative_ratio",
        "neutral_ratio",
        "controversy_score",
        "sentiment_drop_acceleration",
        # Lags and Rolling statistics
        "volume_lag_1",
        "volume_lag_2",
        "volume_lag_3",
        "growth_rate_lag_1",
        "growth_rate_lag_2",
        "engagement_lag_1",
        "controversy_lag_1",
        "rolling_mean_volume_3",
        "rolling_mean_engagement_3",
        "rolling_mean_controversy_3",
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
        is_training: bool = False,
        feature_subset: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extracts and concatenates feature columns into the Combined Feature Vector.
        forecaster_type: "sarima", "sarimax", "both", or "auto"
        feature_subset: Optional explicit list of columns to extract (for Ablation Studies).
        """
        if feature_subset is not None:
            available_social = [col for col in feature_subset if col in df.columns]
        else:
            available_social = [col for col in self.SOCIAL_LISTENING_FEATURES if col in df.columns]

        if forecaster_type == "sarima":
            ts_cols = [col for col in self.SARIMA_FEATURES if col in df.columns]
        elif forecaster_type == "sarimax":
            ts_cols = [col for col in self.SARIMAX_FEATURES if col in df.columns]
        elif forecaster_type == "both":
            ts_cols = [col for col in (self.SARIMA_FEATURES + self.SARIMAX_FEATURES) if col in df.columns]
        else:
            ts_cols = [col for col in (self.SARIMA_FEATURES + self.SARIMAX_FEATURES) if col in df.columns]

        feature_cols = available_social + ts_cols
        X = df[feature_cols].fillna(0.0).values

        if self.scale_features:
            if is_training or not self.fitted_scaler:
                X = self.scaler.fit_transform(X)
                self.fitted_scaler = True
            else:
                X = self.scaler.transform(X)

        return X, feature_cols
