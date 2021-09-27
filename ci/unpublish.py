from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Sequence, Dict

from .lib.config import (
    load_environments,
)
from .lib.ibapi import Instabase


async def unpublish_from_env(sdk: Instabase, package: str, version: str) -> bool:
    """
    Unpublish the package with the given package name and version from a specific Instabase environment,
    specified by an SDK object
    :return: Whether unpublishing was successful
    """
    success = await sdk.unpublish_solution(package, version)
    logger = logging.getLogger(sdk.name)
    if success:
        logger.info("Unpublishing was successful!")
    else:
        logger.error("Unpublishing was not successful.")
    return success


async def unpublish(package: str, version: str) -> Dict[str, bool]:
    """
    Unpublishes the given package and version from every environment in res/environments.yml
    :param package: the name of the package to unpublish
    :param version: the version to unpublish
    :return: Dict[env_name -> success]
    """
    envs = await load_environments()

    sdks = [
        Instabase(
            name=env_name,
            host=env_dict['host'],
            token=env_dict['token'],
            root_path=env_dict['path'],
        )
        for env_name, env_dict in envs.items()
    ]

    results: Sequence[bool] = await asyncio.gather(
        *[unpublish_from_env(sdk, package, version) for sdk in sdks]
    )
    return {k: v for k, v in zip(envs.keys(), results)}


parser = argparse.ArgumentParser(description='Publish layoutlm package')
parser.add_argument(
    '--log-level',
    dest='log_level',
    default='INFO',
    help="DEBUG, INFO, WARNING, ERROR. Defaults to INFO",
)
parser.add_argument(
    '--package',
    dest='package',
    default='ib_layout_lm_trainer',
    help="The name of the package to unpublish",
)
parser.add_argument(
    '--version',
    dest='version',
    help="The version of the package to unpublish",
)

if __name__ == "__main__":
    namespace = parser.parse_args()

    logging.basicConfig(
        level=namespace.log_level,
        format='[%(levelname)s] [%(asctime)s] [%(name)s] (%(module)s.%(funcName)s:%(lineno)d) %(message)s',
    )
    package = namespace.package
    version = namespace.version
    asyncio.run(unpublish(package, version))
