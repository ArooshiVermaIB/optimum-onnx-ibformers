from __future__ import annotations

import argparse
import asyncio
import backoff
import json
import logging
import os
import uuid
import time
from collections import defaultdict
from json import JSONDecodeError
from pathlib import Path
from typing import List, Dict, Any, NamedTuple, Mapping, Optional, Union

from typing_extensions import TypedDict

from .lib.build import zip_project
from .lib.config import (
    load_environments,
    ModelTestConfig,
    PROJECT_ROOT,
    REMOTE_TEMP_ZIP_PATH,
    DATASET_PREFIX,
    load_model_tests,
)
from .lib.ibapi import Instabase
from .lib import utils
from .lib import exceptions

POLLING_INTERVAL = 30
INFERENCE_TIMEOUT = 300  # Allow 5 minutes to run the model in model service


@backoff.on_exception(backoff.expo, exceptions.ContainerRestartException, max_tries=5)
async def run_training_test(
    sdk: Instabase,
    wait_for: asyncio.Task,
    test_name: str,
    root_path: str,
    remote_code_loation: str,
    dataset_path: str,
    test_config: ModelTestConfig,
) -> [bool, str]:
    logger = logging.getLogger(f"{sdk.name}: {test_name}")

    logger.debug("Awaiting code sync")
    await wait_for
    logger.debug("Starting the model training task")

    success = False

    success, job_id = await sdk.start_model_training_task(
        training_script_path=os.path.join(root_path, remote_code_loation),
        model_project_path=os.path.join(dataset_path, DATASET_PREFIX, test_config["model_project_path"]),
        dataset_project_path=os.path.join(dataset_path, DATASET_PREFIX, test_config["dataset_project_path"]),
        base_model="layout-base",
        hyperparams=test_config["config"],
    )

    state = ""
    start_time = time.time()
    done = False
    job_info = {}
    task_metadata = {}
    while time.time() - start_time < test_config["time_limit"]:
        resp = await sdk.get_training_job_status(job_id)
        job_info = None
        if resp and "job" in resp:
            job_info = resp["job"]
        else:
            # if this issue persists, then time out will occur
            logger.info(f"get job status response: {resp}")
            await asyncio.sleep(POLLING_INTERVAL)
            continue

        if job_info.get("status") in ["ERROR", "FAILURE"]:
            # check if it's due to contanier restart
            if job_info.get("message") and job_info.get("message").startswith("Worker received non-pending task"):
                raise exceptions.ContainerRestartException(job_info.get("message"))

            logger.error(f"Server error: {job_info}")
            return False, job_id

        if job_info.get("status") in ["SUCCESS"]:
            done = True
            break

        task_metadata = json.loads(job_info.get("metadata"))
        progress = task_metadata and task_metadata.get("progress")
        if progress:
            logger.info(
                f"message: {job_info.get('message')} | status: {job_info.get('status')} | progress:{progress:.2%}"
            )

        await asyncio.sleep(POLLING_INTERVAL)

    if not done:
        logger.error(f"Test timed out after {test_config['time_limit']} seconds. Last status: {job_info.get('status')}")
        # Don"t run inference test if training timed out
        return success, job_id

    if job_info.get("results"):
        results = job_info.get("results")[0]
        if "error" in results:
            logger.error(f"Server Error: {results['error']}")
            logger.error(f"Stack Trace: {results.get('strace')}")
            return False, job_id  # Fail fast

    evaluation_results = task_metadata.get("evaluation_results")
    if evaluation_results:
        success = utils.do_comparison(evaluation_results, test_config["metrics"], test_name=test_name)
        if success:
            logging.info(f"Test passed: {test_name}")
    else:
        logging.error("evaluation results were not in the status")
        success = False, job_id

    return success, job_id


class PredictionDict(TypedDict):
    avg_confidence: float
    text: str
    words: List["WordPolyDict"]


async def run_inference_test(
    sdk: Instabase,
    wait_for: Optional[asyncio.Task],
    test_name: str,
    dataset_path: str,
    test_config: ModelTestConfig,
) -> bool:
    logger = logging.getLogger(f"{sdk.name}: {test_name}")

    logger.info("Waiting for training test to finish before starting inference test")
    await wait_for
    success, job_id = wait_for.result()
    if not success:
        return False  # fail fast; if training failed, skip inference

    logger.info(f"Running inference test for training job {job_id}")

    model_project_path = Path(os.path.join(dataset_path, DATASET_PREFIX, test_config["model_project_path"]))
    training_job_path = model_project_path / "training_jobs" / job_id

    # we need to publish model first before we can run refiner
    success, model_name, model_version = await _publish_model(sdk, model_project_path, job_id)
    if not success:
        return False

    refiner_prog_path = training_job_path / "refiner_mod" / "prog"
    logger.debug(f"get refiner full path from {refiner_prog_path}")
    refiner_filenames = await sdk.list_directory(str(refiner_prog_path), use_abspath=True)
    # TODO: add filtering to make sure we locate the correct files only here
    # instead of assuming all files under this directory are the same type of files.
    refiner_filename = refiner_filenames[0]
    refiner_path = refiner_prog_path / refiner_filename

    dataset_project_path = Path(os.path.join(dataset_path, DATASET_PREFIX, test_config["dataset_project_path"]))
    dev_input_folder = dataset_project_path / "out_annotations" / "s1_process_files"

    logger.debug(
        f"Unload the model '{test_name}' in case it's already present. "
        "Repeating 10 times to capture multiple potential pods."
    )
    for _ in range(10):
        await sdk.unload_model(model_name)

    prediction_path = training_job_path / "predictions"
    logger.debug(f"get predictions results from training job predictions path: {prediction_path}")
    prediction_filenames = await sdk.list_directory(str(prediction_path), use_abspath=True)
    # TODO: add filtering to make sure we locate the correct files only here
    # instead of assuming all files under this directory are the same type of files.
    prediction_filename = prediction_filenames[0]
    preds = await sdk.read_file(str(prediction_path / prediction_filename), use_abspath=True)
    preds_dict = dict()
    # This has the inference predictions. They should match the output from model service
    try:
        preds_dict: Mapping[str, Mapping[str, PredictionDict]] = json.loads(preds)
    except JSONDecodeError as e:
        logger.error("Exception occured while reading predictions because: " + str(e))
        await sdk.unpublish_solution(model_name, model_version)
        return False

    logger.info(f"Running Refiner {Path(sdk._host) / 'apps/refiner-v5/edit' / refiner_path}")

    # TODO: For now, just run inference against one input record to make sure it doesn't error out
    # We should ideally run against many or all documents to compare
    resp = await sdk.run_refiner(
        program_path=str(refiner_path),
        input_record_keys=[preds_dict["ibdoc_path"] + "-0"],  # Weird thing that we have to add "-0"
    )

    if "job_id" not in resp:
        logger.error(f"Running Refiner failed with error message: {resp}")
        await sdk.unpublish_solution(model_name, model_version)
        return False

    job_id = resp["job_id"]
    start_time = time.time()
    model_result_by_record = {}
    success = True
    # TODO: This timeout should maybe be by test? Not sure...
    while (time.time() - start_time) < INFERENCE_TIMEOUT:
        await asyncio.sleep(POLLING_INTERVAL)
        status = await sdk.get_async_job_status(job_id)
        logger.debug(status)
        success = (
            # Note that model_result_by_record is being mutated by this helper function
            _extract_refiner_results_from_status(logger, status, model_result_by_record)
            and success
        )
        if status.get("state") != "PENDING":
            break

    for prediction in preds_dict["prediction"]["fields"]:
        field_name = prediction["field_name"]
        train_val = prediction["annotations"][0]["value"]
        ibdoc_path = preds_dict["ibdoc_path"]
        preds_by_field = model_result_by_record[ibdoc_path + "-0"]
        inference_val = preds_by_field[field_name]
        if train_val != inference_val:
            logging.error(
                f"For file <{ibdoc_path}> and field <{field_name}>, expected training prediction <{train_val}> "
                f"equal inference prediction <{inference_val}>"
            )
            success = False

    logger.info(f"Inference test {'passed' if success else 'failed'}: {test_name}")

    # TODO We should return something that indicates whether the test failed
    #   (For now, ideally the logs should suffice)

    # TODO move publish/unpublish to a context manager

    await sdk.unpublish_solution(model_name, model_version)
    return success


async def _publish_model(sdk: Instabase, model_project_path: str, job_id: str) -> [bool, str, str]:
    training_job_path = model_project_path / "training_jobs" / job_id

    success, version_id = await sdk.prepare_publish(str(model_project_path), job_id)
    if success:
        content_folder = training_job_path / "artifact"
        success, output_path = await sdk.create_ibsolution(str(content_folder), str(training_job_path))
        if success:
            success, model_name = await sdk.publish_solution(output_path)
            return success, model_name, version_id
    # failure
    return False, None, None


def _extract_refiner_results_from_status(logger, status, model_result_by_record) -> bool:
    status_results_by_record = status.get("results", [{}])[0].get("results_by_record", {})
    failed = False

    for filename, d in status_results_by_record.items():
        refined_phrases = d.get("refined_phrases", [])
        result = [i for i in refined_phrases if i.get("label") == "__model_result"]
        if not result:
            continue
        error = result[0].get("error_msg")
        if error:
            failed = True
            logger.error(f"file <{filename}> failed with error message {error}")
            continue
        output_json = result[0].get("word")
        output_dict = output_json and json.loads(output_json)
        output_text_by_field = {k: " ".join(i[0] for i in v) for k, v in output_dict.items()}
        model_result_by_record[filename] = output_text_by_field
    return not failed


async def sync_and_unzip(sdk: Instabase, contents: bytes, remote_code_loation: str) -> None:
    logger = logging.getLogger(sdk.name)
    # TODO Make sure that locations don"t conflict if multiple users test at once
    logger.debug(f"Deleting remote_code_loation")
    await sdk.delete_file(remote_code_loation, recursive=True)
    logger.debug(f"Writing to REMOTE_TEMP_ZIP_PATH")
    await sdk.write_file(REMOTE_TEMP_ZIP_PATH, contents)
    logger.debug(f"Unzipping REMOTE_TEMP_ZIP_PATH to remote_code_loation")
    await sdk.unzip(zip_filepath=REMOTE_TEMP_ZIP_PATH, destination=remote_code_loation)
    logger.debug(f"Deleting REMOTE_TEMP_ZIP_PATH")
    await sdk.delete_file(REMOTE_TEMP_ZIP_PATH)
    logger.debug(f"Done syncing code")


async def run_tests(train: bool, inference: bool, test_name: Optional[str], test_environment: str) -> None:
    model_tests = await load_model_tests()
    if not model_tests:
        logging.error("No tests found in model_tests.yaml.")
        exit(1)

    if test_name:
        model_tests = {k: v for k, v in model_tests.items() if k == test_name}
        if not model_tests:
            logging.error(f"Test '{test_name}' not found in list of tests {list(model_tests.keys())}")
            exit(1)

    # validate test_environment
    envs = await load_environments()
    if test_environment not in envs:
        logging.error(
            f"test_environment is not one of the environments " f"in 'environments.yaml': {[i for i in envs]}"
        )
        exit(1)

    # create sync_and_unzip task first
    sync_tasks = {}  # We want to keep track of when these are finished to avoid conflicts
    env_config = envs[test_environment]
    sdk = Instabase(
        name=test_environment,
        host=env_config["host"],
        token=env_config["token"],
        root_path=env_config["path"],
    )

    # this solves race condition when run tests concurrently
    remote_code_loation = "ibformers_%s" % uuid.uuid4().hex

    if train:
        zip_bytes = zip_project(PROJECT_ROOT)
        sync_tasks[test_environment] = asyncio.create_task(sync_and_unzip(sdk, zip_bytes, remote_code_loation))
        del zip_bytes

    tasks: List[asyncio.Task] = []

    for test_name, test_config in model_tests.items():
        supported_envs = test_config["env"]
        if test_environment not in supported_envs:
            logging.warning(
                f"test_environment: {test_environment} is not supported by list of "
                f"supported envs in {test_name},"
                f"which are: {[i for i in supported_envs]}"
            )
            continue

        env_config = envs[test_environment]
        sdk = Instabase(
            name=test_environment,
            host=env_config["host"],
            token=env_config["token"],
            root_path=env_config["path"],
        )
        train_task = None
        if train:
            train_task = asyncio.create_task(
                run_training_test(
                    sdk,
                    wait_for=sync_tasks[test_environment],
                    test_name=test_name,
                    root_path=env_config["path"],
                    remote_code_loation=remote_code_loation,
                    dataset_path=env_config["dataset_path"],
                    test_config=test_config,
                )
            )
            tasks.append(train_task)
        if inference:
            inference_task = asyncio.create_task(
                run_inference_test(
                    sdk,
                    wait_for=train_task,
                    test_name=test_name,
                    dataset_path=env_config["dataset_path"],
                    test_config=test_config,
                )
            )
            tasks.append(inference_task)

    results = await asyncio.gather(*tasks)
    if not all(results):
        logging.error("At least one test failed")
        exit(1)


parser = argparse.ArgumentParser(description="Run model training regression tests")
parser.add_argument(
    "--environment",
    dest="environment",
    default="doc-insights-sandbox",
    help="environment tests will be running on",
)

parser.add_argument("--log-level", dest="log_level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")

parser.add_argument("--quiet", dest="quiet", action="store_true", default=False, help="Log more concisely")

parser.add_argument(
    "--test",
    dest="test_name",
    default=None,
    help="If specified, runs only the test with the specified name",
    type=str,
)

parser.add_argument("--no-train", dest="train", action="store_false", help="Don't run training tests")
parser.add_argument("--no-inference", dest="inference", action="store_false", help="Don't run inference tests")

parser.set_defaults(train=True, inference=True)

if __name__ == "__main__":
    namespace = parser.parse_args()

    VERBOSE_FORMAT = "[%(levelname)s] [%(asctime)s] [%(name)s] (%(module)s.%(funcName)s:%(lineno)d) %(message)s"

    logging.basicConfig(
        level=namespace.log_level,
        format=VERBOSE_FORMAT if not namespace.quiet else "%(message)s",
    )

    asyncio.get_event_loop().run_until_complete(
        run_tests(
            train=namespace.train,
            inference=namespace.inference,
            test_name=namespace.test_name,
            test_environment=namespace.environment,
        )
    )
