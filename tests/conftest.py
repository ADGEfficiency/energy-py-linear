import logging
import os


def pytest_configure() -> None:
    disable_logger_names = os.getenv("DISABLE_LOGGERS")

    if disable_logger_names is None:
        disable_loggers = ["energypylinear.optimizer"]
    else:
        disable_loggers = disable_logger_names.split(",")

    for logger_name in disable_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
