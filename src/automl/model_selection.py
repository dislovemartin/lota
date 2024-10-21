import logging
import logging.config

# Configure logging
logging.config.fileConfig('../../monitoring/logging/log_config.yaml', disable_existing_loggers=False)
logger = logging.getLogger('automl.model_selection')

class ModelSelection:
    def __init__(self):
        pass

    def select_best_model(self, models: list, metrics: dict) -> str:
        logger.info(f"Selecting best model based on metrics: {metrics}")
        # Placeholder for model selection logic
        return models[0]

