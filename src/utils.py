import time
import logging
import sys

logger = logging.getLogger(__name__)

def get_unix_seconds() -> int:
    return int(time.time())


def setup_logging(log_level: str) -> None:
    """
    Configure the root logger:
      - stream → stdout
      - format → timestamp, level, logger name, message
      - level  → user-chosen via --log
    """
    # Map string → numeric, e.g. 'DEBUG' → logging.DEBUG
    numeric_level = logging.getLevelName(log_level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level!r}")

    # force=True ensures re-config even if basicConfig() was already called
    logging.basicConfig(
        level=numeric_level,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        force=True,
    )