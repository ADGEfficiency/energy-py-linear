"""Structured logging to console and files."""
import logging
import logging.handlers

from rich.console import Console
from rich.logging import RichHandler

console = Console()

logger = logging.getLogger("default_logger")
logger.setLevel(logging.DEBUG)

rich_handler = RichHandler(
    console=console,
    rich_tracebacks=True,
    show_level=False,
    show_time=False,
    show_path=False,
)
rich_handler.setLevel(logging.INFO)
logger.addHandler(rich_handler)


def set_logging_level(logger, level: int) -> None:
    """Sets the logging level for the logger handlers.

    Args:
        level (int): The new logging level to set.
    """
    for handler in logger.handlers:
        if isinstance(handler, RichHandler):
            handler.setLevel(level * 10)
