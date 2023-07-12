import logging
import os

disable_logger_names = os.getenv("DISABLE_LOGGERS")

# If the environment variable is not set, there's nothing to do
if disable_logger_names is None:
    disable_loggers = ["energypylinear.optimizer"]
else:
    disable_loggers = disable_logger_names.split(",")


def pytest_configure() -> None:
    for logger_name in disable_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
