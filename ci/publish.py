from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from .lib.build import zip_project
from .lib.config import (
    load_environments,
    REMOTE_TEMP_ZIP_PATH,
    abspath,
)
from .lib.ibapi import Instabase, MarketplaceRequestResponse


def version_is_new(
    marketplace_list: MarketplaceRequestResponse, env_name, package_name: str, package_version: str
) -> bool:
    """
    Check whether the given package name and package version is present on the given Instabase environment.
    Returns "true" if the given package name-version pair is *absent* from the environment.
    """
    if marketplace_list['status'] != "OK":
        raise RuntimeError(f"Failed to retrieve marketplace list for environment '{env_name}'")
    apps = marketplace_list['apps']
    published_package_list = [i for i in apps if i['name'] == package_name]
    assert (
        len(published_package_list) < 2
    ), f"Found multiple packages in environment '{env_name}' with name '{package_name}'"
    if len(published_package_list) == 0:
        logging.debug(f"No version of {package_name} found on {env_name}")
        return True
    available_versions = published_package_list[0]['available_versions']
    logging.debug(f"Found available versions {available_versions} for {package_name} in {env_name}")
    return package_version not in available_versions


async def sync_and_publish(
    sdk: Instabase, contents: bytes, package_name: str, package_version: str
) -> None:
    logger = logging.getLogger(sdk.name)
    marketplace_list: MarketplaceRequestResponse = await sdk.load_marketplace_list()
    if not version_is_new(
        marketplace_list,
        env_name=sdk.name,
        package_name=package_name,
        package_version=package_version,
    ):
        logger.info(f"Skipping publishing to {sdk.name}: current version is already present")
        return

    logger.info(f"Publishing {package_name} to {sdk.name}")
    logger.debug(f"Deleting old file at {REMOTE_TEMP_ZIP_PATH}, if exists")
    await sdk.delete_file(REMOTE_TEMP_ZIP_PATH)
    logger.debug(f"Writing zipped code to {REMOTE_TEMP_ZIP_PATH}")
    await sdk.write_file(REMOTE_TEMP_ZIP_PATH, contents)
    logger.debug("Publishing to marketplace (this might take a while!)")
    success = await sdk.publish_solution(REMOTE_TEMP_ZIP_PATH)
    if success:
        logger.info("Publishing was successful!")
    else:
        logger.error(
            f"Something went wrong while publishing {package_name} to {sdk.name}. "
            f"Try again with --log-level=DEBUG"
        )


async def publish(package: str):
    envs = await load_environments()

    package_location = abspath(f'../../{package}')
    package_json_location = Path(package_location).parent / "metadata" / 'package.json'
    logging.debug(f"Loading package.json from '{package_json_location}'")
    with open(package_json_location) as f:
        package_json = json.load(f)
    package_name = package_json['name']
    package_version = package_json['version']
    sdks = [
        Instabase(
            name=env_name,
            host=env_dict['host'],
            token=env_dict['token'],
            root_path=env_dict['path'],
        )
        for env_name, env_dict in envs.items()
    ]
    zip_bytes = zip_project(package_location)
    return await asyncio.gather(
        *[
            sync_and_publish(
                sdk,
                zip_bytes,
                package_name=package_name,
                package_version=package_version,
            )
            for sdk in sdks
        ]
    )


parser = argparse.ArgumentParser(description='Publish ibformers package')
parser.add_argument(
    '--log-level',
    dest='log_level',
    default='INFO',
    help="DEBUG, INFO, WARNING, ERROR. Defaults to INFO",
)
parser.add_argument(
    '--package',
    dest='package',
    default='ibformers',
    help="The location of the package to publish, relative to the ibformers root directory, Defaults to ibformers",
)

if __name__ == "__main__":
    namespace = parser.parse_args()

    logging.basicConfig(
        level=namespace.log_level,
        format='[%(levelname)s] [%(asctime)s] [%(name)s] (%(module)s.%(funcName)s:%(lineno)d) %(message)s',
    )
    package = namespace.package
    asyncio.run(publish(package))
