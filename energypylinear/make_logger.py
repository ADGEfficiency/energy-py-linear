import json
import logging


def make_logger(log_file_path=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter(
            "%(asctime)-15s %(levelname)-8s %(message)s"
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger


def read_logs(log_file_path):
    with open(log_file_path) as f:
        logs = f.read().splitlines()

    return [json.loads(log) for log in logs]

