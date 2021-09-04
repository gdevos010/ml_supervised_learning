import inspect
import logging
import os

logger = logging.getLogger(__name__)


def info(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe())[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    prefix = f'[{os.path.basename(filename)}:{line_number} - {function_name}'.ljust(35) + ']'

    logger.info('{prefix} {i} {m}'.format(
        prefix=prefix,
        i=' ' * max(0, indentation_level - 8),
        m=msg
    ))


def debug(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe())[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logger.debug('{i} {m}'.format(
        i=' ' * max(0, indentation_level - 8),
        m=msg
    ))


def initLogger():
    logger = logging.getLogger(__name__)
    FORMAT = ""
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.DEBUG)
    # log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # logging.basicConfig(level=logging.INFO, format=log_fmt)
