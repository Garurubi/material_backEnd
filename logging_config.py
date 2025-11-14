# logging_config.py
import logging.config
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "access": {
            # uvicorn 기본 포맷과 거의 비슷하게
            "format": "%(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s",
        },
    },
    "handlers": {
        "access_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": str(LOG_DIR / "access.log"),
            "when": "midnight",
            "backupCount": 7,
            "encoding": "utf-8",
            "formatter": "access",
        },
    },
    "loggers": {
        "uvicorn.access": {
            "handlers": ["access_file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
