"""Module for configuring Pytest with custom logger settings.

This module allows users to disable specific loggers when running pytest.
"""

import logging
import os

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def set_pandas_options() -> None:
    """Forces pandas to print all columns on one line."""
    pd.set_option("display.max_columns", 24)
    pd.set_option("display.width", 1000)


def pytest_configure() -> None:
    """Disable specific loggers during pytest runs.

    Loggers to be disabled can be set using the DISABLE_LOGGERS environment variable.

    By default, it disables the logger "energypylinear.optimizer".
    """
    disable_logger_names = os.getenv("DISABLE_LOGGERS")

    if disable_logger_names is None:
        disable_loggers = ["energypylinear.optimizer"]
    else:
        disable_loggers = disable_logger_names.split(",")

    for logger_name in disable_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
