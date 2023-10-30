"""Structured logging to console and files."""
import logging
import logging.handlers

from rich.console import Console
from rich.logging import RichHandler

from energypylinear.defaults import defaults

console = Console()

logger = logging.getLogger("energypylinear")
logger.setLevel(logging.DEBUG)

rich_handler = RichHandler(
    console=console,
    rich_tracebacks=True,
    show_level=False,
    show_time=False,
    show_path=False,
)
rich_handler.setLevel(defaults.log_level * 10)
logger.addHandler(rich_handler)


def set_logging_level(logger: logging.Logger, level: int) -> None:
    """Sets the logging level for the logger handlers.

    Args:
        level (int): The new logging level to set.
    """
    for handler in logger.handlers:
        if isinstance(handler, RichHandler):
            handler.setLevel(level * 10)
