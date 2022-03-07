from __future__ import annotations

import asyncio
import json
import logging
import time
import os
from pathlib import Path
from typing import AnyStr, Dict, Any, List, Optional, Mapping, Union, Tuple

import backoff  # add retry for sync api calls
import aiohttp
from typing_extensions import TypedDict, Literal

from . import exceptions


class Instabase:
    def __init__(self, name: str, host: str, token: str, root_path: str):
        self.name = name
        self._host = host
        self._token = token
        self._root_path = root_path
        self.drive_api_url = os.path.join(self._host, "api/v1", "drives")
        self.logger = logging.getLogger(name)

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=3
    )
    async def start_model_training_task(
        self,
        *,
        training_script_path: str,
        model_project_path: str,
        dataset_project_path: str,
        base_model: str,
        hyperparams: Dict[str, Any],
    ) -> [bool, Union[str, Dict[str, Any], None]]:
        url = os.path.join(self._host, "api/v1/model/training_job")

        arguments = dict(
            model_project_path=model_project_path,
            dataset_paths=[dataset_project_path],
            base_model=base_model,
            hyperparameters=hyperparams,
        )

        hyperparams["training_scripts"] = {"extraction": {"path": training_script_path}}

        self.logger.debug(f"Starting model training job: {arguments}")
        async with aiohttp.ClientSession() as session:
            async with session.put(
                url,
                headers=self._make_headers(),
                data=json.dumps(arguments),
                raise_for_status=True,
            ) as r:
                resp = await r.json()

        if "job_id" in resp:
            return True, resp["job_id"]

        self.logger.error(f"{self.name}: Error while starting model training job: {resp}")
        return False, resp

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=3
    )
    async def run_refiner(
        self,
        program_path: str,
        input_record_keys: List[str],
        use_json: bool = True,
        use_file_cache=True,
        run_with_targets=False,
        refined_phrases: Mapping[str, Any] = None,
    ) -> Dict[str, Any]:
        url = os.path.join(self._host, "api/v1/refiner-v5/run")

        refined_phrases = refined_phrases or {}

        program = await self.read_file(program_path, use_abspath=True)
        path_parts = self._root_path.split("/")
        repo_owner, repo_name = path_parts[0], path_parts[1]
        data = dict(
            program=program,
            use_json=use_json,
            repo_name=repo_name,
            repo_owner=repo_owner,
            input_record_keys=input_record_keys,
            program_path=program_path,
            refined_phrases=refined_phrases,
            use_file_cache=use_file_cache,
            run_with_targets=run_with_targets,
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=self._make_headers(),
                data=json.dumps(data),
                raise_for_status=True,
            ) as r:
                resp = await r.json()
        return resp

    @backoff.on_exception(
        backoff.expo,
        (
            aiohttp.client_exceptions.ClientError,
            aiohttp.client_exceptions.ContentTypeError,
            exceptions.ServerUnavailableException,
        ),
        max_tries=3,
    )
    async def unload_model(self, model_name: str, model_version: str) -> Dict[str, Any]:
        url = os.path.join(self._host, "api/v1/model-service/unload_model")

        self.logger.debug(f"{self.name}: Unloading model with name <{model_name}> and version <{model_version}>")

        data = dict(model_name=model_name, model_version=model_version)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=self._make_headers(),
                data=json.dumps(data),
                raise_for_status=True,
            ) as r:
                resp = await r.json()

        if not resp.get("message") == "Model unloading initiated":
            raise exceptions.ServerUnavailableException(f"Error occurred while unloading model: {resp}")

        return resp

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=3
    )
    async def prepare_publish(self, ib_path: str, job_id: str) -> [bool, Union[str, dict, None]]:
        """parepare training job to be published

        :param ib_path: The location of the model_project_path
        :param  job_id: job_id of training job
        :returns: True, version if operation successful, else False, response message from API
        """

        url = f"{self._host}/api/v1/model/training_job/prep_publish"
        args = dict(model_project_path=ib_path, training_job_id=job_id)
        data = json.dumps(args)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, headers=self._make_headers(), raise_for_status=True) as r:
                resp = await r.json()

        if resp.get("version", "").upper():
            self.logger.debug(f"{self.name}: response was: {resp}")
            return True, resp.get("version")

        self.logger.error(f"{self.name}: Server Error: {resp}")
        return False, resp

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=3
    )
    async def create_ibsolution(
        self, content_folder: str, output_folder: str
    ) -> [bool, Union[str, Dict[str, Any], None]]:
        """creates the ibsolution

        :param content_folder: The location of the model artifact folder
        :param  output_folder: The location of the output folder to save ibsolution file
        :returns: True, solution_path if operation successful, else False, response message from API
        """

        url = f"{self._host}/api/v1/solution/create"
        args = {"async": "true", "content_folder": content_folder, "output_folder": output_folder}
        data = json.dumps(args)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, headers=self._make_headers(), raise_for_status=True) as r:
                resp = await r.json()

        self.logger.info("waiting for creating ibsolution")

        results = None
        if "job_id" in resp:
            results = await self.wait_for_job_completion(resp["job_id"], wait_time=1, is_async=True)

        if results:
            if results["status"] == "OK":
                self.logger.debug(f"{self.name}: response was: {results}")
                return True, results.get("solution_path")

        self.logger.error(f"{self.name}: Server Error: {resp}")
        return False, results

    @backoff.on_exception(
        backoff.expo,
        (
            aiohttp.client_exceptions.ClientError,
            aiohttp.client_exceptions.ContentTypeError,
            exceptions.ServerUnavailableException,
        ),
        max_tries=8,
    )
    async def publish_solution(self, ib_path: str) -> [bool, Union[str, Dict[str, Any], None]]:
        """Publishes the ibsolution located at ib_path on instabase.com to the marketplace

        :param ib_path: The location of the ibsolution on instabase.com
        :returns: True, model_name if operation successful, else False, response message from API
        """

        url = f"{self._host}/api/v1/marketplace/publish"
        args = {"async": "true", "ibsolution_path": ib_path}
        data = json.dumps(args)

        self.logger.info(f"{self.name}: publishing solution found at {ib_path}")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, headers=self._make_headers(), raise_for_status=True) as r:
                resp = await r.json()

        self.logger.info("waiting for publishing ibsolution")

        results = None
        if "job_id" in resp:
            results = await self.wait_for_job_completion(resp["job_id"], wait_time=1, is_async=True)

        if results:
            if results["status"] == "OK":
                self.logger.debug(f"{self.name}: response was: {results}")
                return True, Path(ib_path).name.split("-")[0]  # TODO: find a better way to obtain model_name
            else:
                if ("UNAVAILABLE" in results) or (
                    results and "msg" in results and "UNAVAILABLE" in results["msg"]
                ):  # retry when file service is not available
                    raise exceptions.ServerUnavailableException(results)
                self.logger.error(f"{self.name}: Server Error: {results}")
        return False, results

    @backoff.on_exception(
        backoff.expo,
        (
            aiohttp.client_exceptions.ClientError,
            aiohttp.client_exceptions.ContentTypeError,
            exceptions.ServerUnavailableException,
        ),
        max_tries=8,
    )
    async def unpublish_solution(self, name: str, version: str) -> bool:
        """Unpublishes the ibsolution located at ib_path on instabase.com to the marketplace

        :param ib_path: The location of the ibsolution on instabase.com
        :returns: True, None if operation successful, else False, response message from API
        """

        url = f"{self._host}/api/v1/marketplace/unpublish"
        args = dict(name=name, version=version)
        data = json.dumps(args)

        self.logger.info(f"{self.name}: unpublishing {name} {version}")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, headers=self._make_headers(), raise_for_status=True) as r:
                resp = await r.json()

        if resp.get("status", "").upper() == "OK":
            self.logger.debug(f"{self.name}: response was: {resp}")
            return True

        if "UNAVAILABLE" in results:  # retry when file service is not available
            raise exceptions.ServerUnavailableException(results)

        self.logger.error(f"{self.name}: Server response when unpublishing: {resp}")

        return False

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=3
    )
    async def load_marketplace_list(self) -> MarketplaceRequestResponse:
        url = f"{self._host}/api/v1/marketplace/list"

        self.logger.debug(f"{self.name}: Requesting {url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self._make_headers(), raise_for_status=True) as r:
                resp = await r.json()

        return resp

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=3
    )
    async def read_file(self, ib_path: str, use_abspath: bool = False) -> str:
        headers = {
            **self._make_headers(),
            "Instabase-API-Args": json.dumps(dict(type="file", get_content=True)),
        }

        if use_abspath:
            url = os.path.join(self.drive_api_url, ib_path)
        else:
            url = os.path.join(self.drive_api_url, self._root_path, ib_path)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                raise_for_status=True,
            ) as r:
                resp = await r.text()

        self.logger.debug(f"{self.name} read_file: IB API response head: {resp}")
        return resp

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=3
    )
    async def list_directory(self, ib_path: str, use_abspath: bool = False) -> List[str]:

        if use_abspath:
            url = os.path.join(self.drive_api_url, ib_path)
        else:
            url = os.path.join(self.drive_api_url, self._root_path, ib_path)

        self.logger.info(f"{self.name}: Listing directory {ib_path}")

        headers = {
            **self._make_headers(),
            "Instabase-API-Args": json.dumps(
                dict(type="folder", if_exists="overwrite", get_content=True, get_metadata=False, start_page_token="")
            ),
        }
        has_more = True
        dir_content = []
        while has_more:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    raise_for_status=True,
                ) as r:
                    resp = await r.json()
            dir_content += [r["name"] for r in resp.get("nodes", [])]
            has_more = resp["has_more"]
            if has_more:
                next_page_token = resp["next_page_token"]
                headers["Instabase-API-Args"] = json.dumps(
                    dict(
                        type="folder",
                        if_exists="overwrite",
                        get_content=True,
                        get_metadata=False,
                        start_page_token=next_page_token,
                    )
                )
                self.logger.info(f"{self.name}: Listing further part of the directory {ib_path}")
        return dir_content

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=3
    )
    async def read_binary(self, ib_path: str) -> bytes:
        headers = {
            **self._make_headers(),
            "Instabase-API-Args": json.dumps(dict(type="file", get_content=True)),
        }

        url = os.path.join(self.drive_api_url, self._root_path, ib_path)
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                raise_for_status=True,
            ) as r:
                resp = await r.read()

        self.logger.debug(f"{self.name} read_binary: IB API response: {resp[:100]}")
        return resp

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=3
    )
    async def write_file(self, ib_path: str, contents: AnyStr) -> Dict[str, Any]:
        headers = {
            **self._make_headers(),
            "Instabase-API-Args": json.dumps(dict(type="file", if_exists="overwrite")),
        }
        if isinstance(contents, str):
            contents = contents.encode("utf-8")

        url = os.path.join(self.drive_api_url, self._root_path, ib_path)

        TIMEOUT_SECONDS = 2400  # 40 minutes
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, data=contents, headers=headers, raise_for_status=True, timeout=TIMEOUT_SECONDS
            ) as r:
                resp = await r.json()

        self.logger.debug(f"{self.name} write_file: IB API response: {resp}")
        self.logger.info(f"{self.name} write_file: Wrote to path {ib_path}")
        return resp

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=3
    )
    async def delete_file(self, ib_path: str, *, recursive: bool = False) -> Dict[str, Any]:
        url = os.path.join(self.drive_api_url, self._root_path, ib_path)

        data = json.dumps(dict(force=recursive))
        self.logger.info(f"{self.name}: Deleting file at location {ib_path}")

        TIMEOUT_SECONDS = 2400  # 40 minutes
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                url, data=data, headers=self._make_headers(), raise_for_status=True, timeout=TIMEOUT_SECONDS
            ) as r:
                resp = await r.json()
        self.logger.debug(f"{self.name}: Response was: {resp}")
        return resp

    @backoff.on_exception(
        backoff.expo,
        (
            aiohttp.client_exceptions.ClientError,
            aiohttp.client_exceptions.ContentTypeError,
            exceptions.EmptyResponseException,
        ),
        max_tries=8,
    )
    async def wait_for_job_completion(self, job_id: str, wait_time: float, is_async: bool = False) -> Dict[str, Any]:
        """Repeatedly hits the job status API until a completion/failure signal is received

        :param job_id: The ID of the job to wait for
        :returns: The results returned from the job
        """
        async_arg = "&type=async" if is_async else ""
        url = f"{self._host}/api/v1/jobs/status?job_id={job_id}{async_arg}"

        self.logger.debug(f"{self.name}: Waiting for async job with ID: {job_id}")

        start_time = time.time()
        while time.time() - start_time < 2400:  # TODO: parametrize upper bound of total wait time
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self._make_headers(),
                    raise_for_status=True,
                ) as r:
                    resp = await r.json()
                    if not resp or "state" not in resp:
                        raise exceptions.EmptyResponseException(f"the response is: {resp}")

                    if resp["state"] != "PENDING":
                        return resp["results"][0]

            await asyncio.sleep(wait_time)

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=6
    )
    async def get_training_job_status(self, job_id: str) -> Dict[str, Any]:
        url = os.path.join(self._host, "api/v1/model/training_job") + f"?training_job_id={job_id}"

        self.logger.debug(f"{self.name}: Waiting for async job with ID: {job_id}")

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=self._make_headers(),
                raise_for_status=True,
            ) as r:
                resp = await r.json()

        self.logger.debug(f"{self.name}: async job with ID {job_id} returned with: {resp}")
        return resp

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=3
    )
    async def get_async_job_status(self, job_id: str) -> Dict[str, Any]:
        url = os.path.join(self._host, "api/v1/jobs/status") + f"?job_id={job_id}&type=async"

        self.logger.debug(f"{self.name}: Waiting for async job with ID: {job_id}")

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=self._make_headers(),
                raise_for_status=True,
            ) as r:
                resp = await r.json()

        self.logger.debug(f"{self.name}: async job with ID {job_id} returned with: {resp}")
        return resp

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=3
    )
    async def get_model_training_logs(self, job_id: str) -> List[IClassifierLog]:
        """
        :param job_id: The ID of the job to get logs for
        :return: A list of logs
        """
        url = os.path.join(self._host, "classifier/logs", job_id)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self._make_headers(), raise_for_status=True) as r:
                resp = await r.json()
        return resp

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=3
    )
    async def unzip(self, zip_filepath: str, destination: str) -> bool:
        """Unzips file at zip_filepath on IB into destination. Waits for the
        unzip to finish before returning.

        :param ib_path: string location of the zip file, assumed to be a zip file
        :returns: True if unzip was successful, else returns False
        """

        zip_filename = os.path.basename(zip_filepath)
        zip_dir = os.path.dirname(zip_filepath)

        url = os.path.join(self.drive_api_url, self._root_path, zip_dir, "unzip")

        data = dict(
            zip_file=zip_filename,
            destination=os.path.join(self._root_path, destination),
        )

        api_args = dict(type="folder")

        headers = {
            **self._make_headers(),
            "Instabase-API-Args": json.dumps(api_args),
        }

        self.logger.debug(f"Sending request to unzip '{zip_filename}' to '{destination}'")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=json.dumps(data), headers=headers, raise_for_status=True) as r:
                resp = await r.json()

        self.logger.debug(f"Unzip response while trying to unzip file at {zip_filename}: {resp}")

        if resp.get("status") != "OK":
            msg = resp.get("msg")
            self.logger.error(f"Unzipping returned an error with: {msg}")
            return False

        job_id = resp.get("job_id")
        if not job_id:
            self.logger.error(f"Unzipping returned no job_id")
            return False

        resp = await self.wait_for_job_completion(job_id, wait_time=1, is_async=False)

        if resp.get("status") != "OK":
            msg = resp.get("msg")
            self.logger.error(f"Unzipping returned an error with: {msg}")
            return False

        self.logger.info(f"{self.name}: Unzipped uploaded zip at {zip_filepath} at folder {destination}")
        return True

    @backoff.on_exception(
        backoff.expo, (aiohttp.client_exceptions.ClientError, aiohttp.client_exceptions.ContentTypeError), max_tries=3
    )
    async def run_flow(
        self, input_dir: str, binary_path: str, output_has_run_id: bool = False, delete_out_dir: bool = False
    ) -> Tuple[bool, Optional[Tuple[str, str]]]:
        url = os.path.join(self._host, "api/v1/flow/run_binary_async")
        headers = self._make_headers()
        settings = dict(output_has_run_id=output_has_run_id, delete_out_dir=delete_out_dir)
        data = dict(input_dir=input_dir, binary_path=binary_path, settings=settings)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                data=json.dumps(data),
                raise_for_status=True,
            ) as r:
                resp = await r.json()
        if resp["status"] == "OK":
            return True, (resp["data"]["job_id"], resp["data"]["output_folder"])
        else:
            return False, None

    def _make_headers(self):
        return dict(Authorization=f"Bearer {self._token}")


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
    level: Literal["WARNING", "INFO", "ERROR", "DEBUG"]
    log: str
    user: bool
