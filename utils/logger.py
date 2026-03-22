# utils/logger.py
# ============================================================
#  Centralized logger using loguru
#  loguru is much cleaner than stdlib logging:
#  - Auto-formats with timestamps, level, file/line
#  - Rotates log files automatically
#  - Thread-safe by default
# ============================================================

import sys
import os
from loguru import logger
from utils.config import config

# Remove default loguru handler so we can customise format
logger.remove()

# ---- Console handler (stdout) ----
logger.add(
    sys.stdout,
    level=config.LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# ---- File handler (rotating) ----
os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
logger.add(
    config.LOG_FILE,
    level=config.LOG_LEVEL,
    rotation="10 MB",       # New file every 10 MB
    retention="7 days",     # Keep logs for 7 days
    compression="zip",      # Compress old logs
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
)

# Export the configured logger for import elsewhere:
# from utils.logger import logger
