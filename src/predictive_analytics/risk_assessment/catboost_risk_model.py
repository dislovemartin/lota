from catboost import CatBoostClassifier
import logging
import logging.config

# Configure logging
logging.config.fileConfig('../../../monitoring/logging/log_config.yaml', disable_existing_loggers=False)
logger = logging.getLogger('predictive_analytics.risk_assessment')

class RiskAssessmentModel:
    def __init__(self):
        self.model = CatBoostClassifier()

    def train(self, X, y):
        logger.info("Training CatBoost Risk Assessment Model...")
        self.model.fit(X, y)
        logger.info("Model training completed.")
    
    def predict(self, X):
        logger.info("Predicting risk scores...")
        return self.model.predict_proba(X)[:, 1].tolist()

