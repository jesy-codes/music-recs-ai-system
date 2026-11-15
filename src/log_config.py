"""Logging setup shared by every entry point (CLI, Streamlit app, scripts).

Logs go to the console and to logs/app.log so a demo run can be inspected afterwards.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once; calling it again does nothing."""
    root = logging.getLogger()
    if getattr(root, "_music_recs_configured", False):
        return
    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        LOG_DIR.mkdir(exist_ok=True)
        file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as error:
        # A read-only filesystem (some hosts) shouldn't stop the app; console logging still works.
        root.warning("Could not write log file %s (%s); logging to console only.", LOG_FILE, error)

    # Third-party HTTP clients log every request at INFO; keep our logs readable.
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._music_recs_configured = True
