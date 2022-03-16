import logging
import os


def print_dir(path):
    for dirpath, dirnames, filenames in os.walk(path):
        directory_level = dirpath.replace(path, "")
        directory_level = directory_level.count(os.sep)
        indent = " " * 4
        logging.info("{}{}/".format(indent * directory_level, os.path.basename(dirpath)))

        for f in filenames:
            logging.info("{}{}".format(indent * (directory_level + 1), f))
