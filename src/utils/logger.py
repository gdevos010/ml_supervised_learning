import inspect
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def info(msg):
    level = "INFO"
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe())[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    now = datetime.now().time().strftime("%H:%M:%S")
    prefix = f'[{now} {os.path.basename(filename)}:{line_number} - {function_name}'.ljust(45) + f'] {level}:'

    logger.info('{prefix}{i}\t{m}'.format(
        prefix=prefix,
        i=' ' * max(0, indentation_level - 8),
        m=msg
    ))


def debug(msg):
    level = "DEBUG"
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe())[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    now = datetime.now().time().strftime("%H:%M:%S")
    prefix = f'[{now} {os.path.basename(filename)}:{line_number} - {function_name}'.ljust(45) + f'] {level}:'

    logger.info('{prefix}{i}\t{m}'.format(
        prefix=prefix,
        i=' ' * max(0, indentation_level - 8),
        m=msg
    ))


def error(msg):
    level = "ERROR"
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe())[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    now = datetime.now().time().strftime("%H:%M:%S")
    prefix = f'[{now} {os.path.basename(filename)}:{line_number} - {function_name}'.ljust(45) + f'] {level}:'

    logger.info('{prefix}{i}\t{m}'.format(
        prefix=prefix,
        i=' ' * max(0, indentation_level - 8),
        m=msg
    ))


def init_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="",
                        handlers=[
                            logging.FileHandler("ml_supervised.log", 'a'),
                            logging.StreamHandler()
                        ])
    logger.setLevel(logging.DEBUG)
