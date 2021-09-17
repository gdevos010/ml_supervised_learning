import inspect
import logging
import os

logger = logging.getLogger(__name__)


def info(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe())[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    prefix = f'[{os.path.basename(filename)}:{line_number} - {function_name}'.ljust(30) + '] INFO:'

    logger.info('{prefix}{i}\t{m}'.format(
        prefix=prefix,
        i=' ' * max(0, indentation_level - 8),
        m=msg
    ))


def debug(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe())[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    prefix = f'[{os.path.basename(filename)}:{line_number} - {function_name}'.ljust(30) + '] DEBUG:'

    logger.debug('{prefix}{i}\t{m}'.format(
        prefix=prefix,
        i=' ' * max(0, indentation_level - 8),
        m=msg
    ))


def error(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe())[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    prefix = f'[{os.path.basename(filename)}:{line_number} - {function_name}'.ljust(30) + '] ERROR:'

    logger.error('{prefix} {i} {m}'.format(
        prefix=prefix,
        i=' ' * max(0, indentation_level - 8),
        m=msg
    ))


def init_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="")
    logger.setLevel(logging.DEBUG)
