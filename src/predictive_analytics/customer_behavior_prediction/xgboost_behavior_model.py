import xgboost as xgb
import logging
import logging.config

# Configure logging
logging.config.fileConfig('../../../monitoring/logging/log_config.yaml', disable_existing_loggers=False)
logger = logging.getLogger('predictive_analytics.customer_behavior_prediction')

class BehaviorPredictionModel:
    def __init__(self):
        self.model = xgb.XGBClassifier()

    def train(self, X, y):
        logger.info("Training XGBoost Customer Behavior Prediction Model...")
        self.model.fit(X, y)
        logger.info("Model training completed.")
    
    def predict(self, X):
        logger.info("Predicting customer behavior...")
        return self.model.predict(X).tolist()

