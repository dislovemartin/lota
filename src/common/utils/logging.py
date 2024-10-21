import logging
import logging.config

def setup_logging(default_path='../../monitoring/logging/log_config.yaml', default_level=logging.INFO):
    logging.config.fileConfig(default_path, disable_existing_loggers=False)

