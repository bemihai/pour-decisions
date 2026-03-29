"""Custom logging configuration for Pour Decisions.

Provides a pre-configured ``logger`` instance with colour-coded output,
relative source paths, and log-level control via ``LOG_LEVEL`` environment
variable (0 = NOTSET, 1 = INFO [default], 2 = DEBUG).
"""

import logging
import os
import sys
from pathlib import Path

_log_levels = {
    0: logging.NOTSET,  # no logging
    1: logging.INFO,  # default logging level, prints INFO and above
    2: logging.DEBUG,  # debug logging level, prints DEBUG and above
}


def _get_src_from_path(pathname: str | Path) -> str | Path | None:
    """
    Extract the absolute path to the `src` directory from a given path.
    If `src` is not on the path, return None.
    """
    pathname = Path(pathname).absolute()
    if pathname == Path("/"):
        return None
    if pathname.name == "src":
        return pathname
    return _get_src_from_path(pathname.parent)


class PackagePathFilter(logging.Filter):
    """Custom filter to add the relative path to the log record."""

    def filter(self, record):
        """Add ``relativepath`` attribute to log record for display."""
        pathname = Path(record.pathname)
        src_path = _get_src_from_path(pathname)
        if src_path:
            record.relativepath = pathname.relative_to(src_path)
        else:
            record.relativepath = pathname.name
        return True


class CustomFormatter(logging.Formatter):
    """Custom formatter for the logger."""

    # display colors for log messages
    yellow = "\x1b[33;20m"
    green = "\x1b[32;20m"
    cyan = "\x1b[36;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # log message prefix/postfix
    pre = "[%(asctime)s-%(name)s-%(levelname)s]"
    post = "(%(relativepath)s:%(funcName)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: f"{cyan}{pre}{reset} %(message)s {yellow}{post}{reset}",
        logging.INFO: f"{green}{pre}{reset} %(message)s {yellow}{post}{reset}",
        logging.WARNING: f"{yellow}{pre}{reset} %(message)s {yellow}{post}{reset}",
        logging.ERROR: f"{red}{pre}{reset} %(message)s {yellow}{post}{reset}",
        logging.CRITICAL: f"{bold_red}{pre}{reset} %(message)s {yellow}{post}{reset}",
    }

    def format(self, record):
        """Format the log record with level-specific colour coding."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        formatter.formatTime = self.formatTime
        return formatter.format(record)

    def formatTime(self, record, datefmt=None):
        """Format timestamp as ``YYYY-MM-DD HH:MM:SS``."""
        return super().formatTime(record, "%Y-%m-%d %H:%M:%S")


def build_logger(key: str) -> logging.Logger:
    """
    Returns the custom logger with the specified key.
    The default log level is `INFO`, change it via the `KEY_LOGLEVEL` env variable.
    """
    # build logger
    _logger = logging.getLogger(key)
    # set the log level
    log_level = int(os.environ.get(f"{key}_LOGLEVEL", 1))
    _logger.setLevel(_log_levels[log_level])
    # add custom formatting to the logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    handler.addFilter(PackagePathFilter())
    _logger.addHandler(handler)

    return _logger


# exported module logger
logger = build_logger("POUR-DECISIONS")
