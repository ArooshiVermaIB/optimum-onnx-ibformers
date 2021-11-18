from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import AnyStr, Dict, Any, List, Optional, Mapping

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

    async def read_file(self, ib_path: str) -> str:
        headers = {
            **self._make_headers(),
            'Instabase-API-Args': json.dumps(dict(type='file', get_content=True)),
        }

        url = os.path.join(self.drive_api_url, self._root_path, ib_path)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, ) as r:
                resp = await r.text()

        self.logger.debug(f'{self.name} read_file: IB API response head: {resp}')
        return resp

    async def read_binary(self, ib_path: str) -> bytes:
        headers = {
            **self._make_headers(),
            'Instabase-API-Args': json.dumps(dict(type='file', get_content=True)),
        }

        url = os.path.join(self.drive_api_url, self._root_path, ib_path)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, ) as r:
                resp = await r.read()

        self.logger.debug(f'{self.name} read_binary: IB API response: {resp[:100]}')
        return resp

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

    async def list_directory(self, ib_path: str) -> List[str]:

        url = os.path.join(self.drive_api_url, self._root_path, ib_path)
        self.logger.info(f'{self.name}: Listing directory {ib_path}')

        headers = {
            **self._make_headers(),
            'Instabase-API-Args': json.dumps(dict(type='folder', if_exists='overwrite',
                                                  get_content=True, get_metadata=False,
                                                  start_page_token='')),
        }
        has_more = True
        dir_content = []
        while has_more:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, ) as r:
                    resp = await r.json()
            dir_content += [r['name'] for r in resp.get('nodes', [])]
            has_more = resp['has_more']
            if has_more:
                next_page_token = resp['next_page_token']
                headers['Instabase-API-Args'] = json.dumps(dict(
                    type='folder', if_exists='overwrite', get_content=True,
                    get_metadata=False, start_page_token=next_page_token))
                self.logger.info(f'{self.name}: Listing further part of the directory {ib_path}')
        return dir_content

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
            async with session.get(url, headers=self._make_headers(), ) as r:
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
            script_packages=[os.path.join(self._root_path, script_package)],
            function=script_function,
            arguments=arguments,
            device_type=device_type,
        )
        self.logger.debug(f"Starting model training job: {data}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self._make_headers(), data=json.dumps(data), ) as r:
                resp = await r.json()
        if 'job_id' in resp:
            return resp['job_id']
        self.logger.error(f'{self.name}: Error while starting model training job: {resp}')
        exit(1)

    async def create_refiner_for_model(
            self,
            *,
            model_path: str,
            refiner_path: str,
            model_name: str,
            save_path: str,
            dev_input_folder
    ):
        url = os.path.join(self._host, 'api/v1/annotator/export_model')

        model_path = Path(self._root_path) / model_path

        data = dict(
            model_path=str(model_path) + "/",
            dev_input_folder=dev_input_folder,
            model_name=model_name,
            extracted_fields=[],  # TODO Do we need these?
            refiner_path=str(refiner_path) + "/",  # TODO: ?????
            annotator_dir=str(Path(self._root_path) / save_path) + "/",
        )

        self.logger.debug(f"Creating Refiner")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self._make_headers(), data=json.dumps(data), ) as r:
                resp = await r.json()
        if 'refiner_path' in resp:
            return resp['refiner_path']
        self.logger.error(f'{self.name}: Error while creating Refiner: {resp}')
        exit(1)

    async def run_refiner(self, program_path: str, input_record_keys: List[str], use_json: bool = True,
                          use_file_cache=True, run_with_targets=False, refined_phrases: Mapping[str, Any] = None):
        url = os.path.join(self._host, "api/v1/refiner-v5/run")

        refined_phrases = refined_phrases or {}

        program = await self.read_file(program_path)
        path_parts = self._root_path.split("/")
        repo_owner, repo_name = path_parts[0], path_parts[1]
        data = dict(program=program,
                    use_json=use_json,
                    repo_name=repo_name,
                    repo_owner=repo_owner,
                    input_record_keys=input_record_keys,
                    program_path=os.path.join(self._root_path, program_path),
                    refined_phrases=refined_phrases,
                    use_file_cache=use_file_cache,
                    run_with_targets=run_with_targets)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self._make_headers(), data=json.dumps(data), ) as r:
                resp = await r.json()
        return resp

    async def unload_model(self, model_name: str):
        url = os.path.join(self._host, "api/v1/model-service/unload_model")

        self.logger.debug(f'{self.name}: Unloading model with name <{model_name}>')

        data = dict(model_name=model_name)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self._make_headers(), data=json.dumps(data), ) as r:
                resp = await r.json()

        if not resp.get('message') == "Model unloading initiated":
            self.logger.error(f"Error occurred while unloading model: {resp}")
        return resp

    async def get_async_job_status(self, job_id: str) -> Dict[str, Any]:
        url = os.path.join(self._host, 'api/v1/jobs/status') + f'?job_id={job_id}&type=async'

        self.logger.debug(f'{self.name}: Waiting for async job with ID: {job_id}')

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self._make_headers(), ) as r:
                resp = await r.json()

        self.logger.debug(f'{self.name}: async job with ID {job_id} returned with: {resp}')
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
