from __future__ import annotations

import os
from typing import Dict, Any

import yaml
from typing_extensions import TypedDict


def abspath(relpath):
    dirpath, _ = os.path.split(__file__)
    return os.path.abspath(os.path.join(dirpath, relpath))


# PROJECT_ROOT = abspath('../../test_package')
PROJECT_ROOT = abspath('../../ibformers')
REMOTE_TEMP_ZIP_PATH = 'temp.ibsolution'  # TODO: Make this random
REMOTE_CODE_LOCATION = 'ibformers'  # TODO: Make this within a randomly generated directory
SCRIPT_FUNCTION = 'ibformers.trainer.ib_utils.run_train_annotator'


class Environment(TypedDict):
    host: str
    token: str
    path: str


class ModelTestConfig(TypedDict):
    env: str
    time_limit: float
    ibannotator: str
    config: Dict[str, Any]
    metrics: Dict[str, Dict[str, float]]


async def load_environments() -> Dict[str, Environment]:
    with open(abspath('../res/environments.yml')) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


async def load_model_tests() -> Dict[str, ModelTestConfig]:
    with open(abspath('../res/model_tests.yaml')) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
