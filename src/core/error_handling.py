"""
error_handling.py
=================
Contains functions to handle and log errors that occur during trading operations.
"""

import logging


def handle_error(error_message, exception=None):
    logger = logging.getLogger(__name__)
    if exception:
        logger.error(f"{error_message}: {exception}")
    else:
        logger.error(error_message)
