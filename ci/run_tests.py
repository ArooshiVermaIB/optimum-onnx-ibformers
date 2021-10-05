from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, NamedTuple, Mapping

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
):
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

            if not isinstance(evaluation_results_for_metric[field], float) or \
                    value > evaluation_results_for_metric[field]:
                logger.error(
                    f"for field '{field}' and metric '{metric}' expected at "
                    f"least '{value}' but observed '{evaluation_results_for_metric[field]}'"
                )
                success = False
    return success


async def run_test(
    sdk: Instabase, sync_task: asyncio.Task, test_name: str, test_config: ModelTestConfig
):
    logger = logging.getLogger(test_name)

    logger.debug("Getting mount details")
    mount_details = await sdk.get_mount_details(test_config['ibannotator'])
    logger.debug("Awaiting code sync")
    await sync_task
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
            exit(1)
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
        logger.error(f"Test timed out after {test_config['time_limit']} seconds. Last status: {status}")
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

    # Perform inference test after training, even if the evaluation results were bad
    inference_success = await run_inference_test(sdk, test_name, test_config)
    return success and inference_success


class PredictionDict(TypedDict):
    avg_confidence: float
    text: str
    words: List['WordPolyDict']


async def run_inference_test(sdk: Instabase, test_name: str, test_config: ModelTestConfig):
    logger = logging.getLogger(test_name)

    logger.info("Running inference test")

    dataset_filename = test_config['ibannotator']
    save_path = 'save_location'
    model_name = test_name

    logger.debug("Unload the model_name in case it's already present. "
                 "Repeating 10 times to capture multiple potential pods.")
    for _ in range(10):
        await sdk.unload_model(model_name)

    dataset_path = Path(dataset_filename)
    annotator_dir = dataset_path.parent
    project_name = dataset_path.stem
    model_path = Path(save_path) / 'saved_model'
    refiner_filename = 'inference_test_refiner.ibrefiner'
    dev_input_folder = f"{annotator_dir}/{project_name}_input/out/s2_map_records/"

    logger.info("Creating Refiner")

    refiner_path = await sdk.create_refiner_for_model(
        model_path=str(model_path),
        refiner_path=str(refiner_filename),
        model_name=model_name,
        save_path=save_path,
        dev_input_folder=dev_input_folder
    )

    assert refiner_path, "Refiner path not found"

    preds = await sdk.read_file(str(Path(save_path) / "predictions.json"))

    # This has the inference predictions. They should match the output from model service
    preds_dict: Mapping[str, Mapping[str, PredictionDict]] = json.loads(preds)

    logger.info("Running Refiner")

    # TODO: For now, just run inference against one input record to make sure it doesn't error out
    # We should ideally run against many or all documents to compare
    resp = await sdk.run_refiner(program_path=os.path.join(save_path, refiner_filename),
                                 input_record_keys=[list(preds_dict.keys())[0] + "-0"])
    if 'job_id' not in resp:
        logger.error(f"Running Refiner failed with error message: {resp}")
    job_id = resp['job_id']
    start_time = time.time()
    model_result_by_record = {}
    failed = False
    # TODO: This timeout should maybe be by test? Not sure...
    while (time.time() - start_time) < INFERENCE_TIMEOUT:
        await asyncio.sleep(POLLING_INTERVAL)
        status = await sdk.get_async_job_status(job_id)
        logger.debug(status)
        success = _extract_refiner_results_from_status(logger, status, model_result_by_record)
        failed = failed or not success
        if status.get('state') != "PENDING":
            break

    for filename in model_result_by_record.keys():
        preds_by_field = model_result_by_record[filename]
        logger.info(f"predictions for file <{filename}>: {preds_by_field}")
        # TODO: Instead of printing out the predictions, compare to the preds_dict above!
    logger.info(f"Inference test {'failed' if failed else 'passed'}")

    # TODO We should return something that indicates whether the test failed
    #   (For now, ideally the logs should suffice)


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
        output_text_by_field = {k: " ".join(i[0] for i in v) for k, v in output_dict}
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


async def run_tests():
    model_tests = await load_model_tests()
    envs = await load_environments()
    for test_name, test_config in model_tests.items():
        env = test_config['env']
        if env not in envs:
            logging.error(
                f"In 'model_tests.yaml' for {test_name}, the attribute 'env' is set to '{env}', which is not "
                f"one of the environments in 'environments.yaml': {[i for i in envs]}"
            )
            exit(1)

    sdks = {}

    zip_bytes = zip_project(PROJECT_ROOT)

    sync_tasks = {}

    for env_name in {test_config['env'] for test_config in model_tests.values()}:
        env_config = envs[env_name]
        sdk = Instabase(
            name=env_name,
            host=env_config['host'],
            token=env_config['token'],
            root_path=env_config['path'],
        )
        sync_tasks[env_name] = asyncio.create_task(sync_and_unzip(sdk, zip_bytes))
    del zip_bytes

    tasks: List[asyncio.Task] = []

    for test_name, test_config in model_tests.items():
        env_name = test_config['env']
        env_config = envs[env_name]
        sdk = Instabase(
            name=env_name,
            host=env_config['host'],
            token=env_config['token'],
            root_path=env_config['path'],
        )

        tasks.append(
            asyncio.create_task(
                run_test(
                    sdk,
                    sync_task=sync_tasks[env_name],
                    test_name=test_name,
                    test_config=test_config,
                )
            )
        )

    results = await asyncio.gather(*tasks)
    if not all(results):
        logging.error("At least one test failed")
        exit(1)


parser = argparse.ArgumentParser(description='Run model training regression tests')
parser.add_argument(
    '--log-level', dest='log_level', default='INFO', help="DEBUG, INFO, WARNING, ERROR"
)

if __name__ == "__main__":
    namespace = parser.parse_args()

    logging.basicConfig(
        level=namespace.log_level,
        format='[%(levelname)s] [%(asctime)s] [%(name)s] (%(module)s.%(funcName)s:%(lineno)d) %(message)s',
    )

    asyncio.run(run_tests())
