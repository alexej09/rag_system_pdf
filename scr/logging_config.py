# logging_config.py

import logging
from datetime import datetime

def setup_logging():
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter for log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler for output to terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler for output to a log file
    file_handler = logging.FileHandler('process.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log the start time
    logger.info("Script execution started.")

    return logger

def log_end_time():
    logging.info("Script execution finished.")
