import logging

disable_loggers = ["energypylinear.optimizer"]


def pytest_configure():
    for logger_name in disable_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
