from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import AnyStr, Dict, Any, List, Optional

import aiohttp
from typing_extensions import TypedDict, Literal


class Instabase:
    def __init__(self, name: str, host: str, token: str, root_path: str):
        self.name = name
        self._host = host
        self._token = token
        self._root_path = root_path
        self.drive_api_url = os.path.join(self._host, 'api/v1', 'drives')
        self.logger = logging.getLogger(name)

    def _make_headers(self):
        return dict(Authorization=f'Bearer {self._token}')

    async def write_file(self, ib_path: str, contents: AnyStr) -> Dict[str, Any]:
        headers = {
            **self._make_headers(),
            'Instabase-API-Args': json.dumps(dict(type='file', if_exists='overwrite')),
        }
        if isinstance(contents, str):
            contents = contents.encode('utf-8')

        url = os.path.join(self.drive_api_url, self._root_path, ib_path)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=contents, headers=headers) as r:
                resp = await r.json()

        self.logger.debug(f'{self.name} write_file: IB API response: {resp}')
        self.logger.info(f'{self.name} write_file: Wrote to path {ib_path}')
        return resp

    async def delete_file(self, ib_path: str, *, recursive: bool = False):
        url = os.path.join(self.drive_api_url, self._root_path, ib_path)

        data = json.dumps(dict(force=recursive))
        self.logger.info(f'{self.name}: Deleting file at location {ib_path}')

        async with aiohttp.ClientSession() as session:
            async with session.delete(url, data=data, headers=self._make_headers()) as r:
                resp = await r.json()
        self.logger.debug(f'{self.name}: Response was: {resp}')
        return resp

    async def publish_solution(self, ib_path: str) -> bool:
        """Publishes the ibsolution located at ib_path on instabase.com to the marketplace

        :param ib_path: The location of the ibsolution on instabase.com
        :returns: True, None if operation successful, else False, response message from API
        """

        url = f'{self._host}/api/v1/marketplace/publish'
        args = dict(ibsolution_path=os.path.join(self._root_path, ib_path))
        data = json.dumps(args)

        self.logger.info(f'{self.name}: publishing solution found at {ib_path}')

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, headers=self._make_headers()) as r:
                resp = await r.json()

        if resp.get('status', '').upper() == 'OK':
            self.logger.debug(f'{self.name}: response was: {resp}')
            return True
        self.logger.error(f"{self.name}: Server Error: {resp.get('msg', '')}")
        return False

    async def unpublish_solution(self, name: str, version: str) -> bool:
        """Unpublishes the ibsolution located at ib_path on instabase.com to the marketplace

        :param ib_path: The location of the ibsolution on instabase.com
        :returns: True, None if operation successful, else False, response message from API
        """

        url = f'{self._host}/api/v1/marketplace/unpublish'
        args = dict(name=name, version=version)
        data = json.dumps(args)

        self.logger.info(f'{self.name}: unpublishing {name} {version}')

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, headers=self._make_headers()) as r:
                resp = await r.json()

        if resp.get('status', '').upper() == 'OK':
            self.logger.debug(f'{self.name}: response was: {resp}')
            return True
        self.logger.error(f"{self.name}: Server response when unpublishing: {resp}")
        return False

    async def unzip(self, zip_filepath: str, destination: str) -> bool:
        """Unzips file at zip_filepath on IB into destination. Waits for the
        unzip to finish before returning.

        :param ib_path: string location of the zip file, assumed to be a zip file
        :returns: True if unzip was successful, else returns False
        """

        zip_filename = os.path.basename(zip_filepath)
        zip_dir = os.path.dirname(zip_filepath)

        url = os.path.join(self.drive_api_url, self._root_path, zip_dir, 'unzip')

        data = dict(
            zip_file=zip_filename,
            destination=os.path.join(self._root_path, destination),
        )

        api_args = dict(type='folder')

        headers = {
            **self._make_headers(),
            'Instabase-API-Args': json.dumps(api_args),
        }

        self.logger.debug(f"Sending request to unzip '{zip_filename}' to '{destination}'")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=json.dumps(data), headers=headers) as r:
                resp = await r.json()

        self.logger.debug(f'Unzip response while trying to unzip file at {zip_filename}: {resp}')

        if resp.get('status') != 'OK':
            msg = resp.get('msg')
            self.logger.error(f'Unzipping returned an error with: {msg}')
            return False

        job_id = resp.get('job_id')
        if not job_id:
            self.logger.error(f'Unzipping returned no job_id')
            return False

        resp = await self.wait_for(job_id, wait_time=1)  # TODO

        if resp.get('status') != 'OK':
            msg = resp.get('msg')
            self.logger.error(f'Unzipping returned an error with: {msg}')
            return False

        self.logger.info(
            f'{self.name}: Unzipped uploaded zip at {zip_filepath} at folder {destination}'
        )
        return True

    async def load_marketplace_list(self) -> MarketplaceRequestResponse:
        url = f"{self._host}/api/v1/marketplace/list"

        self.logger.debug(f"{self.name}: Requesting {url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self._make_headers()) as r:
                resp = await r.json()

        return resp

    async def wait_for(self, job_id: str, *, wait_time: float) -> Dict[str, Any]:
        """Repeatedly hits the job status API until a completion/failure signal is received

        :param job_id: The ID of the job to wait for
        :returns: The results returned from the job
        """
        url = os.path.join(self._host, 'api/v1/jobs/status') + f'?job_id={job_id}'

        self.logger.info(f'{self.name}: Waiting for job with ID: {job_id}')

        while True:

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._make_headers()) as r:
                    resp = await r.json()

            self.logger.debug(f'{self.name}: Job with ID {job_id} returned with: {resp}')

            if resp['status'] != 'OK':
                # Raise?
                return resp

            if resp['state'] != 'PENDING':
                return resp

            await asyncio.sleep(wait_time)

    async def get_mount_details(self, dataset_location: str) -> Optional[MountDetails]:
        i = dataset_location.find('/fs/')
        if i == -1:
            self.logger.debug(
                f"{self.name}: Couldn't find /fs/ in the dataset_location '{dataset_location}'"
            )
            return None
        root = dataset_location[: i + 3]
        url = os.path.join(self.drive_api_url, root)
        logging.debug(f"Sending request to {url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self._make_headers()) as r:
                resp = await r.json()
        drives = resp.get('drives', [])
        if not drives:
            self.logger.debug(f"{self.name}: Nothing in resp.drives. Response was {resp}")
            return None
        drives = [
            i
            for i in drives
            if i.get('full_path') and dataset_location.startswith(i.get('full_path'))
        ]
        if not drives:
            self.logger.debug(
                f"{self.name}: Couldn't find the drive for dataset_location '{dataset_location}'"
            )
            return None
        assert 'mount_details' in drives[0], "drive did not have key 'mount_details'"
        return drives[0]['mount_details']

    async def start_model_training_task(
        self,
        *,
        script_package,
        script_function,
        dataset_filename,
        save_path: str,
        mount_details,
        model_name: str,
        hyperparams: Dict[str, Any],
        device_type: Optional[Literal['cpu', 'gpu']] = None,
    ):
        url = os.path.join(self._host, 'api/v1/model-service/run_train')
        arguments = dict(
            dataset_filename=dataset_filename,
            hyperparams=hyperparams,
            save_path=os.path.join(self._root_path, save_path),
            mount_details=mount_details,
            model_name=model_name,
        )
        if not device_type:
            device_type = 'gpu' if hyperparams.get('use_gpu', True) else 'false'
        data = dict(
            script_packages=[os.path.join(self._root_path, script_package, 'src')],
            function=script_function,
            arguments=arguments,
            device_type=device_type,
        )
        self.logger.debug(f"Starting model training job: {data}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self._make_headers(), data=json.dumps(data)) as r:
                resp = await r.json()
        if 'job_id' in resp:
            return resp['job_id']
        self.logger.error(f'{self.name}: Error while starting model training job: {resp}')
        exit(1)

    async def get_model_training_job_status(self, job_id: str) -> Dict[str, Any]:
        url = os.path.join(self._host, 'api/v1/jobs/status') + f'?job_id={job_id}&type=async'

        self.logger.debug(f'{self.name}: Waiting for model training job with ID: {job_id}')

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self._make_headers()) as r:
                resp = await r.json()

        self.logger.debug(f'{self.name}: Model training job with ID {job_id} returned with: {resp}')
        return resp

    async def get_model_training_logs(self, job_id: str) -> List[IClassifierLog]:
        """
        :param job_id: The ID of the job to get logs for
        :return: A list of logs
        """
        url = os.path.join(self._host, 'classifier/logs', job_id)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self._make_headers()) as r:
                resp = await r.json()
        return resp


class MarketplaceRequestResponse(TypedDict):
    status: str
    apps: List[MarketplaceAppResponse]


class MarketplaceAppResponse(TypedDict):
    name: str
    available_versions: List[str]
    installed_versions: List[str]
    open_urls: Dict[str, str]
    src_paths: Dict[str, str]


class MountDetails(TypedDict):
    client_type: str  # should be S3
    prefix: str
    bucket_name: str
    aws_region: str


class IClassifierLog(TypedDict):
    ts: str
    level: Literal['WARNING', 'INFO', 'ERROR', 'DEBUG']
    log: str
    user: bool
