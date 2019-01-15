
import json
import logging


def make_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('log.log')
    stream_handler = logging.StreamHandler()

    stream_formatter = logging.Formatter(
            "%(asctime)-15s %(levelname)-8s %(message)s"
    )

    file_formatter = logging.Formatter('%(message)s')

    file_handler.setFormatter(file_formatter)
    stream_handler.setFormatter(stream_formatter)

    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def read_logs():
    with open('log.log') as f:
        logs = f.read().splitlines()

    return [json.loads(log) for log in logs]

