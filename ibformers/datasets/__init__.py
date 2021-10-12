import os


def _abspath():
    dirpath, _ = os.path.split(__file__)
    return dirpath


DATASETS_PATH = _abspath()
