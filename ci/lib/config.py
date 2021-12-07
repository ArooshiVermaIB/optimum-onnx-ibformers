from __future__ import annotations

import os
import uuid
from typing import Dict, Any

import yaml
from typing_extensions import TypedDict


def abspath(relpath):
    dirpath, _ = os.path.split(__file__)
    return os.path.abspath(os.path.join(dirpath, relpath))


PROJECT_ROOT = abspath("../../ibformers")
REMOTE_TEMP_ZIP_PATH = "%s.ibsolution" % uuid.uuid4().hex
REMOTE_CODE_LOCATION = "ibformers"  # TODO: Make this within a randomly generated directory
REMOTE_CODE_PREFIX = "ci-test-temp"


def load_environments() -> Dict[str, Dict]:
    with open(abspath("../res/environments.yml")) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_model_tests() -> Dict[str, Dict]:
    with open(abspath("../res/model_tests.yaml")) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_model_config() -> Dict[str, Dict]:
    with open(abspath("../res/base_models.yaml")) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
