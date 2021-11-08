from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from json import JSONDecodeError
from pathlib import Path
from typing import List, Dict, Any, NamedTuple, Mapping, Optional

from typing_extensions import TypedDict

from .lib.build import zip_project
from .lib.config import ModelTestConfig, PROJECT_ROOT, SCRIPT_FUNCTION
from .lib.config import (
    load_environments,
    REMOTE_TEMP_ZIP_PATH,
    REMOTE_CODE_LOCATION,
    load_model_tests,
)
from .lib.ibapi import Instabase

POLLING_INTERVAL = 10
INFERENCE_TIMEOUT = 300  # Allow 5 minutes to run the model in model service


def do_comparison(
    evaluation_results: Dict[str, Dict[str, Any]],
    expected_metrics: Dict[str, Dict[str, Any]],
    *,
    test_name: str,
) -> bool:
    """
    Check that the evaluation_results are each at least as high as the expected metrics
    >>> evaluation_results = {
    >>>     "f1_score": {
    >>>         "name": 0.95,
    >>>         "address": 0.9,
    >>>     }, "accuracy": {
    >>>         "name": 0.6,
    >>>         "address": 0.7,
    >>>     }, "pecision": {
    >>>         "name": 0.7,
    >>>         "address": 0.8,
    >>>     }
    >>> }
    >>> expected_metrics = {
    >>>        "f1_score": {
    >>>         "name": 0.9
    >>>     }, "accuracy": {
    >>>         "name": 0.5
    >>>     }
    >>> }
    >>> do_comparison(evaluation_results, expected_metrics)
    True
    """
    logger = logging.getLogger(test_name)
    success = True
    for metric, fields in expected_metrics.items():
        if metric not in evaluation_results:
            logger.error(f"evaluation results did not contain expected metric {metric}")
            success = False
            continue
        evaluation_results_for_metric = evaluation_results[metric]
        for field, value in fields.items():
            if field not in evaluation_results_for_metric:
                logger.error(
                    f"evaluation results did not contain expected field {field} for metric {metric}"
                )
                success = False
                continue
            observed = evaluation_results_for_metric[field]
            try:
                observed = float(observed)
            except ValueError:
                logger.error(
                    f"for field '{field}' and metric '{metric}' observed value '{observed}' "
                    f"could not be converted to a float"
                )
                success = False
                continue

            if (
                not isinstance(evaluation_results_for_metric[field], float)
                or value > evaluation_results_for_metric[field]
            ):
                logger.error(
                    f"for field '{field}' and metric '{metric}' expected at "
                    f"least '{value}' but observed '{evaluation_results_for_metric[field]}'"
                )
                success = False
    return success


async def run_training_test(
    sdk: Instabase, wait_for: asyncio.Task, test_name: str, test_config: ModelTestConfig
):
    logger = logging.getLogger(test_name)

    logger.debug("Getting mount details")
    mount_details = await sdk.get_mount_details(test_config['ibannotator'])
    logger.debug("Awaiting code sync")
    await wait_for
    logger.debug("Starting the model training task")
    save_path = f'save_location/{test_name}'

    job_id = await sdk.start_model_training_task(
        script_package=REMOTE_CODE_LOCATION + "/ibformers",  # TODO change this!!!
        script_function=SCRIPT_FUNCTION,  # TODO: Make more generic
        dataset_filename=test_config['ibannotator'],
        save_path=save_path,
        mount_details=mount_details,
        model_name=test_name,
        hyperparams=test_config['config'],
    )
    state = ''
    start_time = time.time()
    success = False
    done = False
    status = {}
    task_data = {}
    while time.time() - start_time < test_config['time_limit']:
        status = await sdk.get_async_job_status(job_id)
        # TODO: Add assertions/error messages
        if status.get('status') == "ERROR":
            logger.error(f"Server error: {status}")
            return False
        cur_status = json.loads(status.get('cur_status', '{}'))
        if state != cur_status.get('task_state'):
            logger.info(f"Job state is now {cur_status.get('task_state')}")
        task_data = cur_status.get('task_data')
        progress = task_data and task_data.get('progress')
        if progress:
            logger.info(f'progress:{progress:.2%}')

        state = cur_status.get('task_state')
        if state in {'DONE', 'ERROR', 'FAILURE', 'SUCCESS'} or status.get('status') == "DONE":
            done = True
            break
        await asyncio.sleep(POLLING_INTERVAL)

    if not done:
        logger.error(
            f"Test timed out after {test_config['time_limit']} seconds. Last status: {status}"
        )
        # Don't run inference test if training timed out
        return success

    if status.get('results'):
        results = status.get('results')[0]
        if 'error' in results:
            logger.error(f"Server Error: {results['error']}")
            logger.error(f"Stack Trace: {results.get('strace')}")
            return False  # Fail fast

    evaluation_results = task_data.get('evaluation_results')
    if evaluation_results:
        success = do_comparison(evaluation_results, test_config['metrics'], test_name=test_name)
        if success:
            logging.info("Test passed")
    else:
        logging.error("evaluation results were not in the status")
        success = False

    return success


class PredictionDict(TypedDict):
    avg_confidence: float
    text: str
    words: List['WordPolyDict']


async def run_inference_test(
    sdk: Instabase, wait_for: Optional[asyncio.Task], test_name: str, test_config: ModelTestConfig
) -> bool:
    logger = logging.getLogger(test_name)

    if wait_for:
        logger.info("Waiting for training test to finish before starting inference test")
        await wait_for

    logger.info("Running inference test")

    dataset_filename = test_config['ibannotator']
    save_path = f'save_location/{test_name}'
    model_name = test_name

    logger.debug(
        f"Unload the model '{model_name}' in case it's already present. "
        "Repeating 10 times to capture multiple potential pods."
    )
    for _ in range(10):
        await sdk.unload_model(model_name)

    dataset_path = Path(dataset_filename)
    annotator_dir = dataset_path.parent
    project_name = dataset_path.stem
    model_path = Path(save_path) / 'saved_model'
    refiner_filename = 'inference_test_refiner.ibrefiner'
    dev_input_folder = f"{annotator_dir}/{project_name}_input/out/s2_map_records/"

    logger.info("Creating Refiner")

    refiner_path = str(refiner_filename)

    refiner_path = await sdk.create_refiner_for_model(
        model_path=str(model_path),
        refiner_path=str(refiner_filename),
        model_name=model_name,
        save_path=save_path,
        dev_input_folder=dev_input_folder,
    )

    assert refiner_path, "Refiner path not found"

    preds = await sdk.read_file(str(Path(save_path) / "predictions.json"))

    # This has the inference predictions. They should match the output from model service
    try:
        preds_dict: Mapping[str, Mapping[str, PredictionDict]] = json.loads(preds)
    except JSONDecodeError as e:
        logger.error("Exception occured while reading predictions")
        return False

    logger.info(f"Running Refiner {Path(sdk._host) / 'apps/refiner-v5/edit' / refiner_path}")

    # TODO: For now, just run inference against one input record to make sure it doesn't error out
    # We should ideally run against many or all documents to compare
    resp = await sdk.run_refiner(
        program_path=os.path.join(save_path, refiner_filename),
        input_record_keys=[
            list(preds_dict.keys())[0] + "-0"
        ],  # Weird thing that we have to add "-0"
    )
    if 'job_id' not in resp:
        logger.error(f"Running Refiner failed with error message: {resp}")
    job_id = resp['job_id']
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
        if status.get('state') != "PENDING":
            break

    for filename in model_result_by_record.keys():
        preds_by_field = model_result_by_record[filename]
        training_pred = preds_dict.get(filename[:-2])

        for field, inference_val in preds_by_field.items():
            train_val = training_pred.get(field, {}).get("text", "")
            if inference_val != train_val:
                logging.error(
                    f"For file <{filename}> and field <{field}>, expected training prediction <{train_val}> "
                    f"equal inference prediction <{inference_val}>"
                )
                success = False
        # TODO: Instead of printing out the predictions, compare to the preds_dict above!
    logger.info(f"Inference test {'passed' if success else 'failed'}")

    # TODO We should return something that indicates whether the test failed
    #   (For now, ideally the logs should suffice)

    return success


def _extract_refiner_results_from_status(logger, status, model_result_by_record):
    status_results_by_record = status.get('results', [{}])[0].get("results_by_record", {})
    failed = False
    for filename, d in status_results_by_record.items():
        refined_phrases = d.get("refined_phrases", [])
        result = [i for i in refined_phrases if i.get('label') == '__model_result']
        if not result:
            continue
        error = result[0].get('error_msg')
        if error:
            failed = True
            logger.error(f"file <{filename}> failed with error message {error}")
            continue
        output_json = result[0].get('word')
        output_dict = output_json and json.loads(output_json)
        output_text_by_field = {k: " ".join(i[0] for i in v) for k, v in output_dict.items()}
        model_result_by_record[filename] = output_text_by_field
    return not failed


async def sync_and_unzip(sdk, contents):
    logger = logging.getLogger(sdk.name)
    # TODO Make sure that locations don't conflict if multiple users test at once
    logger.debug(f"Deleting REMOTE_CODE_LOCATION")
    await sdk.delete_file(REMOTE_CODE_LOCATION, recursive=True)
    logger.debug(f"Writing to REMOTE_TEMP_ZIP_PATH")
    await sdk.write_file(REMOTE_TEMP_ZIP_PATH, contents)
    logger.debug(f"Unzipping REMOTE_TEMP_ZIP_PATH to REMOTE_CODE_LOCATION")
    await sdk.unzip(zip_filepath=REMOTE_TEMP_ZIP_PATH, destination=REMOTE_CODE_LOCATION)
    logger.debug(f"Deleting REMOTE_TEMP_ZIP_PATH")
    await sdk.delete_file(REMOTE_TEMP_ZIP_PATH)
    logger.debug(f"Done syncing code")


async def run_tests(train: bool, inference: bool, test_name: Optional[str], test_environment: str):
    model_tests = await load_model_tests()
    if not model_tests:
        logging.error("No tests found in model_tests.yaml.")
        exit(1)

    if test_name:
        model_tests = {k: v for k, v in model_tests.items() if k == test_name}
        if not model_tests:
            logging.error(
                f"Test '{test_name}' not found in list of tests {list(model_tests.keys())}"
            )
            exit(1)

    # validate test_environment
    envs = await load_environments()
    if test_environment not in envs:
        logging.error(
            f"test_environment is not one of the environments "
            f"in 'environments.yaml': {[i for i in envs]}"
        )
        exit(1)

    # create sync_and_unzip task first
    sync_tasks = {}  # We want to keep track of when these are finished to avoid conflicts
    env_config = envs[test_environment]
    sdk = Instabase(
        name=test_environment,
        host=env_config['host'],
        token=env_config['token'],
        root_path=env_config['path'],
    )
    if train:
        zip_bytes = zip_project(PROJECT_ROOT)
        sync_tasks[test_environment] = asyncio.create_task(sync_and_unzip(sdk, zip_bytes))
        del zip_bytes

    tasks: List[asyncio.Task] = []

    for test_name, test_config in model_tests.items():
        supported_envs = test_config['env']
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
            host=env_config['host'],
            token=env_config['token'],
            root_path=env_config['path'],
        )
        train_task = None
        if train:
            train_task = asyncio.create_task(
                run_training_test(
                    sdk,
                    wait_for=sync_tasks[test_environment],
                    test_name=test_name,
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
                    test_config=test_config,
                )
            )
            tasks.append(inference_task)

    results = await asyncio.gather(*tasks)
    if not all(results):
        logging.error("At least one test failed")
        exit(1)


parser = argparse.ArgumentParser(description='Run model training regression tests')
parser.add_argument(
    '--environment',
    dest='environment',
    default='doc-insights-sandbox',
    help="environment tests will be running on",
)

parser.add_argument(
    '--log-level', dest='log_level', default='INFO', help="DEBUG, INFO, WARNING, ERROR"
)

parser.add_argument(
    '--quiet', dest='quiet', action='store_true', default=False, help="Log more concisely"
)

parser.add_argument(
    '--test',
    dest='test_name',
    default=None,
    help="If specified, runs only the test with the specified name",
    type=str,
)

parser.add_argument(
    '--no-train', dest='train', action='store_false', help="Don't run training tests"
)
parser.add_argument(
    '--no-inference', dest='inference', action='store_false', help="Don't run inference tests"
)

parser.set_defaults(train=True, inference=True)

if __name__ == "__main__":
    namespace = parser.parse_args()

    VERBOSE_FORMAT = (
        '[%(levelname)s] [%(asctime)s] [%(name)s] (%(module)s.%(funcName)s:%(lineno)d) %(message)s'
    )

    logging.basicConfig(
        level=namespace.log_level,
        format=VERBOSE_FORMAT if not namespace.quiet else "%(message)s",
    )

    asyncio.run(
        run_tests(
            train=namespace.train,
            inference=namespace.inference,
            test_name=namespace.test_name,
            test_environment=namespace.environment,
        )
    )
