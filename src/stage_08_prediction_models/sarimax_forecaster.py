import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from loguru import logger
from typing import Dict, Any, Tuple, List, Optional

class SARIMAXForecaster:
    """
    Topic Volume Forecasting with Exogenous Regressors (SARIMAX).
    Extends seasonal ARIMA by incorporating multimodal social listening covariates:
    - engagement: Weighted interaction momentum (likes, shares, comments)
    - avg_sentiment: Multilingual sentiment polarity score
    - hashtag_activity: Viral tagging intensity
    - unique_users: Participant diversity

    Topic Volume History + Exogenous Signals -> Future Topic Volume Forecast
    Generates:
    - sarimax_forecast_volume: Projected future discussion volume
    - sarimax_forecast_growth: Projected future growth rate
    """

    DEFAULT_EXOG_COLS = [
        "engagement",
        "avg_sentiment",
        "hashtag_activity",
        "unique_users"
    ]

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.order = tuple(self.config.get("order", (1, 1, 1)))
        self.seasonal_order = tuple(self.config.get("seasonal_order", (1, 1, 0, 7)))
        self.forecast_steps = self.config.get("forecast_steps", 3)
        self.exog_cols = self.config.get("exog_cols", self.DEFAULT_EXOG_COLS)

    def _prepare_exog(self, exog_history: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Cleans exogenous matrix, removes zero-variance columns to avoid multicollinearity,
        and projects future exogenous steps using recent moving averages.
        """
        if exog_history is None or exog_history.size == 0:
            return None, None

        # Filter out NaN or constant columns
        valid_cols = []
        for col_idx in range(exog_history.shape[1]):
            col_vals = exog_history[:, col_idx]
            if not np.all(np.isnan(col_vals)) and np.nanstd(col_vals) > 1e-6:
                valid_cols.append(col_idx)

        if not valid_cols:
            return None, None

        clean_exog = np.nan_to_num(exog_history[:, valid_cols], nan=0.0)

        # Project future exogenous values using the mean of the last min(3, len) observations
        lookback = min(3, len(clean_exog))
        recent_exog_mean = np.mean(clean_exog[-lookback:], axis=0)
        future_exog = np.tile(recent_exog_mean, (self.forecast_steps, 1))

        return clean_exog, future_exog

    def fit_and_forecast_series(
        self,
        history_series: np.ndarray,
        exog_history: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Fits SARIMAX model on historical volume series with exogenous regressors.
        Falls back to univariate SARIMA or moving average if fitting fails or data is insufficient.
        """
        min_obs = max(14, self.seasonal_order[3] * 2 if len(self.seasonal_order) > 3 else 10)
        if len(history_series) < min_obs:
            recent_mean = float(np.mean(history_series[-3:])) if len(history_series) > 0 else 0.0
            return recent_mean * self.forecast_steps, 0.0

        window_size = 40
        window_series = history_series[-window_size:]

        clean_exog, future_exog = None, None
        if exog_history is not None and len(exog_history) == len(history_series):
            window_exog_raw = exog_history[-len(window_series):]
            clean_exog, future_exog = self._prepare_exog(window_exog_raw)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Try SARIMAX with exogenous variables
            if clean_exog is not None and future_exog is not None:
                try:
                    model = SARIMAX(
                        window_series,
                        exog=clean_exog,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    model_fit = model.fit(disp=False, maxiter=25)
                    forecast = model_fit.forecast(steps=self.forecast_steps, exog=future_exog)
                    forecast_vol = float(np.maximum(0.0, np.sum(forecast)))

                    current_vol = window_series[-1]
                    baseline = max(current_vol * self.forecast_steps, 1.0)
                    forecast_growth = float((forecast_vol - baseline) / baseline)
                    return forecast_vol, forecast_growth
                except Exception:
                    pass  # Fall back to univariate SARIMA

            # Fallback 1: Univariate SARIMA
            try:
                model = SARIMAX(
                    window_series,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_fit = model.fit(disp=False, maxiter=20)
                forecast = model_fit.forecast(steps=self.forecast_steps)
                forecast_vol = float(np.maximum(0.0, np.sum(forecast)))

                current_vol = window_series[-1]
                baseline = max(current_vol * self.forecast_steps, 1.0)
                forecast_growth = float((forecast_vol - baseline) / baseline)
                return forecast_vol, forecast_growth
            except Exception:
                # Fallback 2: Moving average extrapolation
                recent_mean = float(np.mean(history_series[-3:]))
                return recent_mean * self.forecast_steps, 0.0

    def generate_forecast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates SARIMAX Forecast Feature columns (sarimax_forecast_volume, sarimax_forecast_growth).
        """
        df = df.sort_values(by=["topic_id", "time_bucket"]).copy()
        avail_exog = [c for c in self.exog_cols if c in df.columns]
        
        forecast_dfs = []
        for topic_id, group in df.groupby("topic_id"):
            vols = group["topic_volume"].values
            exog_matrix = group[avail_exog].values if avail_exog else None
            n = len(vols)
            
            top_fc_vols = np.zeros(n)
            top_fc_growths = np.zeros(n)

            for i in range(n):
                history_vol = vols[: i + 1]
                history_exog = exog_matrix[: i + 1] if exog_matrix is not None else None
                fc_vol, fc_growth = self.fit_and_forecast_series(history_vol, history_exog)
                top_fc_vols[i] = fc_vol
                top_fc_growths[i] = fc_growth

            group = group.copy()
            group["sarimax_forecast_volume"] = top_fc_vols
            group["sarimax_forecast_growth"] = top_fc_growths
            forecast_dfs.append(group)

        result_df = pd.concat(forecast_dfs).sort_values(by=["topic_id", "time_bucket"]).reset_index(drop=True)
        return result_df
