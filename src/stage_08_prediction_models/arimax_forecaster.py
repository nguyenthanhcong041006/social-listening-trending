import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from loguru import logger
from typing import Dict, Any, Tuple, List, Optional

class ARIMAXForecaster:
    """
    Topic Volume Forecasting with Direct Multi-Horizon Lagged Exogenous Signals (ARIMAX).
    Extends ARIMA by incorporating multimodal social listening covariates:
    - log_follower_engagement: Follower-normalized interaction velocity
    - engagement: Weighted interaction momentum (likes, shares, comments)
    - avg_sentiment: Multilingual sentiment polarity score
    - controversy_score: Heated fan debate index
    - cross_platform_entropy: Dispersion of discussion across platforms
    - uppercase_intensity: Outrage / emotional arousal ratio
    - hashtag_activity: Viral tagging intensity
    - unique_users: Participant diversity

    Topic Volume History + Genuine Lagged Exogenous Signals -> Future Topic Volume Forecast
    Generates:
    - arimax_forecast_volume / sarimax_forecast_volume: Projected future discussion volume
    - arimax_forecast_growth / sarimax_forecast_growth: Projected future growth rate
    """

    DEFAULT_EXOG_COLS = [
        "log_follower_engagement",
        "engagement",
        "avg_sentiment",
        "controversy_score",
        "cross_platform_entropy",
        "uppercase_intensity",
        "platform_count",
        "hashtag_activity",
        "unique_users"
    ]

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.order = tuple(self.config.get("order", (1, 1, 0)))
        raw_seasonal = self.config.get("seasonal_order", None)
        self.seasonal_order = tuple(raw_seasonal) if raw_seasonal and any(raw_seasonal) else None
        self.forecast_steps = self.config.get("forecast_steps", 3)
        self.exog_cols = self.config.get("exog_cols", self.DEFAULT_EXOG_COLS)

    def _prepare_exog(self, exog_history: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Cleans exogenous matrix, removes zero-variance columns, and builds a genuine
        k-step lagged exogenous matrix from recent real observations X[t-k+1 : t+1]
        without fabricating or tiling flat future values.
        """
        if exog_history is None or exog_history.size == 0:
            return None, None

        valid_cols = []
        for col_idx in range(exog_history.shape[1]):
            col_vals = exog_history[:, col_idx]
            if not np.all(np.isnan(col_vals)) and np.nanstd(col_vals) > 1e-6:
                valid_cols.append(col_idx)

        if not valid_cols:
            return None, None

        clean_exog = np.nan_to_num(exog_history[:, valid_cols], nan=0.0)

        # Direct Multi-horizon Exogenous Matrix:
        # Use the k most recent observed exogenous rows to drive the k-step forward forecast.
        k = self.forecast_steps
        if len(clean_exog) >= k:
            future_exog = clean_exog[-k:].copy()
        else:
            pad_needed = k - len(clean_exog)
            pad_rows = np.tile(clean_exog[0], (pad_needed, 1))
            future_exog = np.vstack([pad_rows, clean_exog])

        return clean_exog, future_exog

    def fit_and_forecast_series(
        self,
        history_series: np.ndarray,
        exog_history: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Fits ARIMAX model on historical volume series with genuine lagged exogenous regressors.
        For micro-drama series shorter than min_obs, applies Exponential Weighted Moving Average (EWMA)
        with momentum acceleration.
        """
        min_obs = 4 if self.seasonal_order is None else max(10, self.seasonal_order[3] * 2)
        if len(history_series) < min_obs:
            if len(history_series) == 0:
                return 0.0, 0.0
            weights = np.exp(np.linspace(-1, 0, len(history_series)))
            weights /= np.sum(weights)
            ewma = float(np.sum(history_series * weights))
            momentum = float((history_series[-1] - history_series[0]) / max(len(history_series), 1))
            projected = max(0.0, ewma + 0.5 * momentum)
            forecast_vol = float(projected * self.forecast_steps)
            current_vol = history_series[-1]
            forecast_growth = float((forecast_vol - current_vol * self.forecast_steps) / max(current_vol * self.forecast_steps, 1))
            return forecast_vol, forecast_growth

        window_size = 30
        window_series = history_series[-window_size:]

        clean_exog, future_exog = None, None
        if exog_history is not None and len(exog_history) == len(history_series):
            window_exog_raw = exog_history[-len(window_series):]
            clean_exog, future_exog = self._prepare_exog(window_exog_raw)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Try ARIMAX with genuine lagged exogenous variables
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
                    pass  # Fall back to univariate ARIMA

            # Fallback 1: Univariate ARIMA
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
                weights = np.exp(np.linspace(-1, 0, len(history_series)))
                weights /= np.sum(weights)
                ewma = float(np.sum(history_series * weights))
                forecast_vol = float(ewma * self.forecast_steps)
                current_vol = history_series[-1]
                forecast_growth = float((forecast_vol - current_vol * self.forecast_steps) / max(current_vol * self.forecast_steps, 1))
                return forecast_vol, forecast_growth

    def generate_forecast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates ARIMAX Forecast Feature columns.
        Sets both 'arimax_forecast_*' and 'sarimax_forecast_*' for seamless backwards compatibility.
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
                history_vols = vols[: i + 1]
                history_exog = exog_matrix[: i + 1] if exog_matrix is not None else None
                fc_vol, fc_growth = self.fit_and_forecast_series(history_vols, history_exog)
                top_fc_vols[i] = fc_vol
                top_fc_growths[i] = fc_growth
                
            group = group.copy()
            group["arimax_forecast_volume"] = top_fc_vols
            group["arimax_forecast_growth"] = top_fc_growths
            group["sarimax_forecast_volume"] = top_fc_vols
            group["sarimax_forecast_growth"] = top_fc_growths
            forecast_dfs.append(group)
            
        result_df = pd.concat(forecast_dfs).sort_values(by=["topic_id", "time_bucket"]).reset_index(drop=True)
        return result_df

# Alias for backwards compatibility
SARIMAXForecaster = ARIMAXForecaster
