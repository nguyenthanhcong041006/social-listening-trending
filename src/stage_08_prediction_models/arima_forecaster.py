import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from loguru import logger
from typing import Dict, Any, Tuple

class ARIMAForecaster:
    """
    Sub-stage 8.1: Topic Volume Forecasting (ARIMA / Short-Horizon Forecasting)
    Uses non-seasonal Autoregressive Integrated Moving Average (ARIMA) with EWMA
    momentum smoothing to forecast discussion volume time-series for micro-drama sports topics:
        Topic Volume History -> Future Topic Volume Forecast
    
    Generates Forecast Features:
    - arima_forecast_volume / sarima_forecast_volume: Estimated future discussion volume
    - arima_forecast_growth / sarima_forecast_growth: Projected future growth rate
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.order = tuple(self.config.get("order", (1, 1, 0)))
        raw_seasonal = self.config.get("seasonal_order", None)
        self.seasonal_order = tuple(raw_seasonal) if raw_seasonal and any(raw_seasonal) else None
        self.forecast_steps = self.config.get("forecast_steps", 3)

    def fit_and_forecast_series(self, history_series: np.ndarray) -> Tuple[float, float]:
        """
        Fits ARIMA model on the historical series of a topic and forecasts future steps.
        For micro-drama series shorter than min_obs, applies Exponential Weighted Moving Average (EWMA)
        with momentum acceleration instead of crashing.
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
        
        try:
            window_series = history_series[-30:]
            model = SARIMAX(
                window_series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False, maxiter=25)
            forecast = model_fit.forecast(steps=self.forecast_steps)
            forecast_vol = float(max(0.0, np.sum(forecast)))
            
            current_vol = window_series[-1]
            forecast_growth = float((forecast_vol - current_vol * self.forecast_steps) / max(current_vol * self.forecast_steps, 1))
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
        Generates ARIMA Forecast Feature columns across the dataset.
        Sets both 'arima_forecast_*' and 'sarima_forecast_*' for seamless compatibility.
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
            group["arima_forecast_volume"] = top_fc_vols
            group["arima_forecast_growth"] = top_fc_growths
            group["sarima_forecast_volume"] = top_fc_vols
            group["sarima_forecast_growth"] = top_fc_growths
            forecast_vols.append(group)
            
        result_df = pd.concat(forecast_vols).sort_values(by=["topic_id", "time_bucket"]).reset_index(drop=True)
        return result_df

# Alias for backwards compatibility
SARIMAForecaster = ARIMAForecaster
