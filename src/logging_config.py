import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(name: str = "pdf_qa_api") -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    info_handler = RotatingFileHandler(
        "logs/api_info.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    info_handler.setFormatter(formatter)
    info_handler.setLevel(logging.INFO)

    error_handler = RotatingFileHandler(
        "logs/api_error.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)

    logger.addHandler(console_handler)
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    return logger
