import asyncio
import json
import logging
from pathlib import Path
import os
from unittest import TestCase
import sys

sys.path.append("..")  # Adds higher directory to python modules path.

from ci.lib.build import zip_project, PackageType
from ci.lib.ibapi import Instabase
from ci.lib.config import load_environments, abspath
from ci.publish import sync_and_publish, version_is_new


class TestPublish(TestCase):
    def setUp(self) -> None:
        logging.basicConfig(
            level="DEBUG",
            format="[%(levelname)s] [%(asctime)s] [%(name)s] (%(module)s.%(funcName)s:%(lineno)d) %(message)s",
        )

    def test_publish(self):
        async def publish(env_name: str):
            envs = load_environments()

            package = "ibformers_testing"
            package_location = abspath(f"../test/fixtures/test_package/ibformers")
            package_json_location = Path(package_location).parent / f"ci/package_config/{package}.json"
            logging.debug(f"Loading package.json from '{package_json_location}'")
            with open(package_json_location, "r") as f:
                package_json = json.load(f)
            package_name = package_json["name"]
            package_version = package_json["version"]
            self.assertIn(env_name, envs, msg=f"Couldn't find '{env_name}' in environments.yaml")
            env = envs[env_name]
            sdk = Instabase(
                name=env_name,
                host=env["host"],
                token=env["token"],
                root_path=env["path"],
            )
            zip_bytes = zip_project(package_location, package)
            await sync_and_publish(
                sdk,
                "ib_annotation/data/fs/Prod Drive/datasets/ci-test-datasets-do-not-delete",
                zip_bytes,
                package_name=package_name,
                package_version=package_version,
            )

            # Now check it was successful
            marketplace_list = await sdk.load_marketplace_list()
            is_new = version_is_new(
                marketplace_list,
                env_name=sdk.name,
                package_name=package_name,
                package_version=package_version,
            )
            self.assertFalse(is_new, "Package should be in marketplace now")

            success = await sdk.unpublish_solution(package_name, package_version)
            self.assertTrue(success, msg="Unpublishing was not successful")

            marketplace_list = await sdk.load_marketplace_list()
            is_new = version_is_new(
                marketplace_list,
                env_name=sdk.name,
                package_name=package_name,
                package_version=package_version,
            )

            self.assertTrue(is_new, "Package should be in marketplace now")

        asyncio.run(publish("doc-insights-sandbox"))
