from fbprophet import Prophet
import pandas as pd
import logging
import logging.config

# Configure logging
logging.config.fileConfig('../../../monitoring/logging/log_config.yaml', disable_existing_loggers=False)
logger = logging.getLogger('predictive_analytics.time_series_forecasting')

class TimeSeriesForecastingModel:
    def __init__(self):
        self.model = Prophet()

    def train(self, df: pd.DataFrame):
        logger.info("Training Prophet Time Series Forecasting Model...")
        self.model.fit(df)
        logger.info("Model training completed.")
    
    def predict(self, periods: int):
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict(orient='records')

