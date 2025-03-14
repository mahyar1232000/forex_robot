import logging
from datetime import datetime


def setup_logger():
    """Set up logging."""
    logging.basicConfig(
        filename=f"logs/forex_robot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def log_message(message):
    """Log a message."""
    logging.info(message)
