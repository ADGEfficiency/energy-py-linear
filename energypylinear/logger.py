"""Structured logging to console and files."""
import logging
import logging.handlers
import os
import pathlib
import time

import structlog
from rich.console import Console
from rich.logging import RichHandler

console = Console()


class PulpRedirectHandler(logging.Handler):
    """Custom logging handler for redirecting PuLP logs to structlog.

    Attributes:
        _structlog (structlog.BoundLogger): The structlog logger instance.
    """

    def __init__(self, level: int = logging.NOTSET) -> None:
        """Initialize the PulpRedirectHandler with the given logging level.

        Args:
            level (int, optional): The logging level. Defaults to logging.NOTSET.
        """
        super().__init__(level)
        self._structlog = structlog.get_logger()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by formatting and sending it to structlog.

        Args:
            record (logging.LogRecord): The log record to emit.
        """
        message = self.format(record)
        self._structlog.debug("pulp", pulp_message=message)


def configure_logger(enable_file_logging: bool = False) -> None:
    """Configure root and PuLP loggers. Optionally enable file logging.

    Args:
        enable_file_logging (bool, optional):
            Whether to enable file logging. Defaults to False.
    """

    unix_time = int(time.time())

    # Set up the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    if enable_file_logging:
        pathlib.Path("logs").mkdir(exist_ok=True)
        # Set up the file handler to write debug logs to a file
        file_handler = logging.handlers.TimedRotatingFileHandler(
            f"logs/{unix_time}.log", when="midnight", backupCount=10
        )
        file_handler.setLevel(logging.DEBUG)

        latest_file_handler = logging.FileHandler("logs/LATEST.log")
        latest_file_handler.setLevel(logging.DEBUG)

        root_logger.addHandler(file_handler)
        root_logger.addHandler(latest_file_handler)

    # Set up the stream handler to write info logs to stdout
    stream_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_level=False,
        show_time=False,
        show_path=False,
    )
    stream_handler.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)
    pulp_logger = logging.getLogger("pulp")

    # Now set up PuLP to use the redirect handler
    pulp_logger.addHandler(PulpRedirectHandler())
    pulp_logger.propagate = False

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(
                fmt="iso"
            ),  # Add the timestamp to the log entry
            structlog.stdlib.filter_by_level,  # Decide which log levels to process
            structlog.stdlib.add_logger_name,  # Add the logger name to the log entry
            structlog.stdlib.add_log_level,  # Add the log level to the log entry
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,  # Render exception information
            structlog.processors.UnicodeDecoder(),  # Make sure all input is unicode
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,  # Format the log entry
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "default_logger") -> structlog.BoundLogger:
    """Get a logger instance by name, using structlog.

    Args:
        name (str, optional): The name of the logger to retrieve.
        Defaults to "default_logger".

    Returns:
        structlog.BoundLogger: The logger instance.
    """
    return structlog.get_logger(name)


configure_logger(enable_file_logging=bool(os.environ.get("ENABLE_FILE_LOGGING", False)))


logger = get_logger()
