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


async def unpublish(package: str, version: str, env_name: str) -> Dict[str, bool]:
    """
    Unpublishes the given package and version from every environment in res/environments.yml
    :param package: the name of the package to unpublish
    :param version: the version to unpublish
    :return: Dict[env_name -> success]
    """
    envs = load_environments()

    env_dict = dict(envs).get(env_name)
    if env_dict is None:
        logger.error(f"the environment {env_name} does not exist!")
        exit(1)

    sdk = Instabase(
        name=env_name,
        host=env_dict["host"],
        token=env_dict["token"],
        root_path=env_dict["path"],
    )

    return await unpublish_from_env(sdk, package, version)


parser = argparse.ArgumentParser(description="Publish layoutlm package")
parser.add_argument("--environment", dest="environment", help="environment package will be published to")

parser.add_argument(
    "--log-level",
    dest="log_level",
    default="INFO",
    help="DEBUG, INFO, WARNING, ERROR. Defaults to INFO",
)
parser.add_argument(
    "--package",
    dest="package",
    default="ibformers_extraction",
    help="The name of the package to unpublish",
)
parser.add_argument(
    "--version",
    dest="version",
    help="The version of the package to unpublish",
)

if __name__ == "__main__":
    namespace = parser.parse_args()

    logging.basicConfig(
        level=namespace.log_level,
        format="[%(levelname)s] [%(asctime)s] [%(name)s] (%(module)s.%(funcName)s:%(lineno)d) %(message)s",
    )
    package = namespace.package
    version = namespace.version
    env_name = namespace.environment
    asyncio.run(unpublish(package, version, env_name))
