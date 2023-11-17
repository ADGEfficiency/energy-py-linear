"""Logging to console."""
import logging
import logging.handlers

from rich.console import Console
from rich.logging import RichHandler

from energypylinear.defaults import defaults

console = Console(width=80)

logger = logging.getLogger("energypylinear")
logger.setLevel(logging.DEBUG)

rich_handler = RichHandler(
    console=console,
    rich_tracebacks=True,
    show_level=True,
    show_time=False,
    show_path=False,
)
rich_handler.setLevel(defaults.log_level * 10)
logger.addHandler(rich_handler)


def set_logging_level(logger: logging.Logger, level: int | bool) -> None:
    """Sets the logging level for the logger handlers.

    Args:
        level (int): The new logging level to set.
    """
    if isinstance(level, bool):
        if level is True:
            level = defaults.log_level
        else:
            # error
            level = 4

    for handler in logger.handlers:
        if isinstance(handler, RichHandler):
            if level < 10:
                level = level * 10

            handler.setLevel(level)
