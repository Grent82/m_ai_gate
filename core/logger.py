"""Logger utilities for the project."""

import logging
import os
from typing import Optional, Union


def setup_logger(
    name: str,
    log_level: Optional[Union[str, int]] = None,
    enable_file_handler: bool = True,
) -> logging.Logger:
    """Create and configure a :class:`logging.Logger` instance.

    Parameters
    ----------
    name:
        Name of the logger to configure.
    log_level:
        Optional log level. If ``None`` the ``LOG_LEVEL`` environment variable
        is consulted, defaulting to ``"INFO"`` when unset or invalid.
    enable_file_handler:
        Whether to attach a :class:`logging.FileHandler`. When ``False`` only a
        console handler is added.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """

    # Determine the numeric log level
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")

    if isinstance(log_level, str):
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = log_level

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Common formatter for all handlers
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler always enabled
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Optional file handler
    if enable_file_handler and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = logging.FileHandler("agent.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
