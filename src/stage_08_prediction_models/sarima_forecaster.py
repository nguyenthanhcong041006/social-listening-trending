import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from loguru import logger
from typing import Dict, Any, Tuple

class SARIMAForecaster:
    """
    Sub-stage 8.1: Topic Volume Forecasting (SARIMA)
    Uses Seasonal Autoregressive Integrated Moving Average (SARIMA) to forecast
    future discussion volume time-series from historical volume trajectories:
        Topic Volume Current / History -> Future Topic Volume Forecast
    
    Generates SARIMA Forecast Features:
    - sarima_forecast_volume: Estimated future discussion volume
    - sarima_forecast_growth: Projected future growth rate
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.order = tuple(self.config.get("order", (1, 1, 1)))
        self.seasonal_order = tuple(self.config.get("seasonal_order", (1, 1, 0, 7)))
        self.forecast_steps = self.config.get("forecast_steps", 3)

    def fit_and_forecast_series(self, history_series: np.ndarray) -> Tuple[float, float]:
        """
        Fits SARIMA model on the historical series of a topic and forecasts future steps.
        Falls back to moving average extrapolation if historical series is too short or non-convergent.
        """
        min_obs = max(14, self.seasonal_order[3] * 2 if len(self.seasonal_order) > 3 else 10)
        if len(history_series) < min_obs:
            recent_mean = float(np.mean(history_series[-3:])) if len(history_series) > 0 else 0.0
            return recent_mean * self.forecast_steps, 0.0
        
        try:
            # Use recent window up to 40 steps for fast, stable SARIMA fitting
            window_series = history_series[-40:]
            model = SARIMAX(
                window_series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False, maxiter=20)
            forecast = model_fit.forecast(steps=self.forecast_steps)
            forecast_vol = float(np.sum(forecast))
            
            # Forecasted growth rate compared to current baseline
            current_vol = window_series[-1]
            forecast_growth = float((forecast_vol - current_vol * self.forecast_steps) / max(current_vol * self.forecast_steps, 1))
            return forecast_vol, forecast_growth
        except Exception as e:
            recent_mean = float(np.mean(history_series[-3:]))
            return recent_mean * self.forecast_steps, 0.0

    def generate_forecast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates SARIMA Forecast Feature columns across the dataset.
        """
        df = df.sort_values(by=["topic_id", "time_bucket"]).copy()
        forecast_vols = []
        
        for topic_id, group in df.groupby("topic_id"):
            vols = group["topic_volume"].values
            n = len(vols)
            top_fc_vols = np.zeros(n)
            top_fc_growths = np.zeros(n)
            
            for i in range(n):
                history = vols[: i + 1]
                fc_vol, fc_growth = self.fit_and_forecast_series(history)
                top_fc_vols[i] = fc_vol
                top_fc_growths[i] = fc_growth
                
            group = group.copy()
            group["sarima_forecast_volume"] = top_fc_vols
            group["sarima_forecast_growth"] = top_fc_growths
            forecast_vols.append(group)
            
        result_df = pd.concat(forecast_vols).sort_values(by=["topic_id", "time_bucket"]).reset_index(drop=True)
        return result_df
