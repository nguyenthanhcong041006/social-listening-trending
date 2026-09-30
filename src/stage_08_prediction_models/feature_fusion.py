import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from loguru import logger

class FeatureFusion:
    """
    Sub-stage 8.2: Feature Fusion
    Fuses two independent feature streams into a unified representation:
    1. Social Listening Features:
       - topic_volume, growth_rate, engagement, avg_sentiment, sentiment_change,
         hashtag_activity, hashtag_growth, acceleration, unique_users
       - Lag & rolling statistics
    2. SARIMA Forecast Features:
       - sarima_forecast_volume, sarima_forecast_growth
    
    Output: Combined Feature Vector fed directly into the LightGBM classifier.
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

    def __init__(self, scale_features: bool = False):
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.combined_feature_names = self.SOCIAL_LISTENING_FEATURES + self.SARIMA_FEATURES

    def fuse(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Extracts and concatenates feature columns into the Combined Feature Vector.
        """
        available_social = [col for col in self.SOCIAL_LISTENING_FEATURES if col in df.columns]
        available_sarima = [col for col in self.SARIMA_FEATURES if col in df.columns]
        
        feature_cols = available_social + available_sarima
        X = df[feature_cols].fillna(0.0).values
        
        if self.scale_features:
            if is_training:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
                
        logger.debug(f"Fused {X.shape[1]} features ({len(available_social)} Social + {len(available_sarima)} SARIMA).")
        return X, feature_cols
