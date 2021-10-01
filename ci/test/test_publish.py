import asyncio
import json
import logging
import os

from ..lib.build import zip_project
from ..lib.ibapi import Instabase
from ..lib.config import load_environments, abspath
from unittest import TestCase

from ..publish import sync_and_publish, version_is_new


class TestPublish(TestCase):
    def setUp(self) -> None:
        logging.basicConfig(
            level='DEBUG',
            format='[%(levelname)s] [%(asctime)s] [%(name)s] (%(module)s.%(funcName)s:%(lineno)d) %(message)s',
        )

    def test_publish(self):
        async def publish(env_name: str):
            envs = await load_environments()

            package_location = abspath(f'../../test_package')
            package_json_location = os.path.join(package_location, 'package.json')
            logging.debug(f"Loading package.json from '{package_json_location}'")
            with open(package_json_location, 'r') as f:
                package_json = json.load(f)
            package_name = package_json['name']
            package_version = package_json['version']
            self.assertIn(env_name, envs, msg=f"Couldn't find '{env_name}' in environments.yaml")
            env = envs[env_name]
            sdk = Instabase(
                name=env_name,
                host=env['host'],
                token=env['token'],
                root_path=env['path'],
            )
            zip_bytes = zip_project(package_location)
            await sync_and_publish(
                sdk,
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

        asyncio.run(publish('doc-insights-sandbox'))
