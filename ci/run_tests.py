from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from json import JSONDecodeError
from pathlib import Path
from typing import List, Dict, Mapping, Optional

import backoff
from typing_extensions import TypedDict

from .lib import exceptions
from .lib import utils
from .lib.build import zip_project, PackageType
from .lib.config import (
    load_environments,
    PROJECT_ROOT,
    REMOTE_TEMP_ZIP_PATH,
    REMOTE_CODE_PREFIX,
    PUBLISH_LOCK,
    load_model_tests,
)
from .lib.ibapi import Instabase

POLLING_INTERVAL = 30
INFERENCE_TIMEOUT = 300  # Allow 5 minutes to run the model in model service


@backoff.on_exception(backoff.expo, exceptions.ContainerRestartException, max_tries=5)
async def run_training_test(
    sdk: Instabase,
    wait_for: asyncio.Task,
    test_name: str,
    root_path: str,
    remote_code_loation: str,
    test_config: Dict,
    package: str,
) -> [bool, str]:
    logger = logging.getLogger(f"{sdk.name}: {test_name}")

    logger.debug("Awaiting code sync")
    await wait_for
    logger.debug("Starting the model training task")

    success = False
    hyperparams = test_config["config"]
    if package == "ibformers_extraction":
        hyperparams["log_metrics_to_metadata"] = True

    success, job_id = await sdk.start_model_training_task(
        training_script_path=os.path.join(root_path, REMOTE_CODE_PREFIX, remote_code_loation),
        model_project_path=os.path.join(root_path, test_config["model_project_path"]),
        dataset_project_path=os.path.join(root_path, test_config["dataset_project_path"]),
        base_model=utils.get_base_model_name(test_config["config"]),
        hyperparams=hyperparams,
        package=package,
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
        return False, job_id

    return success, job_id


class PredictionDict(TypedDict):
    avg_confidence: float
    text: str
    words: List["WordPolyDict"]


CLASSIFIER_TEMPLATE = {
    "rootDir": "..",
    "steps": [
        {
            "kwargs": {
                "input_folder": "input",
                "output_folder": "out/s1_apply_classifier",
                "classifier_path": "classifier_mod/model_split_classifier_model_0f1d3b1828df43a2951b0ef850d84561.ibclassifier",
                "settings": {"filter_classes": "", "split_inputs": "no"},
            },
            "option": {"internalName": "apply_classifier", "name": "Apply Classifier"},
        }
    ],
}


async def run_inference_test_classifier(
    sdk: Instabase,
    wait_for: Optional[asyncio.Task],
    test_name: str,
    dataset_path: str,
    test_config: Dict,
) -> bool:
    logger = logging.getLogger(f"{sdk.name}: {test_name}")

    logger.info("Waiting for training test to finish before starting inference test")
    await wait_for
    success, job_id = wait_for.result()
    if not success:
        return False  # fail fast; if training failed, skip inference

    logger.info(f"Running inference test for training job {job_id}")

    model_project_path = Path(os.path.join(dataset_path, test_config["model_project_path"]))
    training_job_path = model_project_path / "training_jobs" / job_id

    # we need to publish model first before we can run refiner
    success, model_name, model_version = await _publish_model(sdk, model_project_path, job_id)
    if not success:
        return False

    classifier_scripts_path = training_job_path / "classifier_mod"
    logger.debug(f"get classifier full path from {classifier_scripts_path}")
    classifier_filenames = await sdk.list_directory(str(classifier_scripts_path), use_abspath=True)
    # TODO: add filtering to make sure we locate the correct files only here
    # instead of assuming all files under this directory are the same type of files.
    classifier_filename = [file for file in classifier_filenames if file.endswith(".ibclassifier")][0]
    classifier_path = classifier_scripts_path / classifier_filename

    dataset_project_path = Path(os.path.join(dataset_path, test_config["dataset_project_path"]))
    dev_input_folder = dataset_project_path / "out_annotations" / "s1_process_files"

    logger.debug(
        f"Unload the model '{test_name}' in case it's already present. "
        "Repeating 10 times to capture multiple potential pods."
    )
    for _ in range(10):
        await sdk.unload_model(model_name, model_version)

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

    logger.info(f"Running classifier {Path(sdk._host) / 'apps/classifier' / classifier_path}")

    inference_uuid = uuid.uuid4().hex

    classifier_flow_path = training_job_path / f"ci_test_flow_{inference_uuid}"
    await sdk.make_dir(str(classifier_flow_path), use_abspath=True)
    CLASSIFIER_TEMPLATE["steps"][0]["kwargs"]["classifier_path"] = str(Path("classifier_mod") / classifier_filename)
    flow_path = classifier_flow_path / "ci_test.ibflow"

    logger.info(f"Creating classifier flow: {flow_path}")
    await sdk.write_file(str(flow_path), json.dumps(CLASSIFIER_TEMPLATE), use_abspath=True)

    ci_input_folder = training_job_path / f"ci_input_{inference_uuid}"
    logger.info(f"Creating input folder for classifier flow: {ci_input_folder}")
    await sdk.make_dir(str(ci_input_folder), use_abspath=True)

    input_ibdoc_name = os.path.basename(preds_dict["ibdoc_path"])
    task_type = test_config["config"]["task_type"]
    if task_type == "classification":
        copy_resp = await sdk.copy_file(
            preds_dict["ibdoc_path"], str(ci_input_folder / input_ibdoc_name), use_abspath=True
        )
    elif task_type == "split_classification":
        copy_resp = await sdk.copy_file(
            str(dev_input_folder / "original" / input_ibdoc_name),
            str(ci_input_folder / input_ibdoc_name),
            use_abspath=True,
        )

    await sdk.wait_for_job_completion(copy_resp["job_id"], 1)

    logger.info(f"Running classsifier flow {flow_path} on input folder {ci_input_folder}")
    success, resp = await sdk.run_flow_test("/" + str(ci_input_folder), "/" + str(flow_path), output_has_run_id=True)

    if not success:
        logger.error(f"Running classifier flow failed with error message: {resp}")
        await sdk.unpublish_solution(model_name, model_version)
        return False

    job_id = resp["data"]["job_id"]
    out_folder = resp["data"]["output_folder"]

    await sdk.wait_for_job_completion(job_id, POLLING_INTERVAL)

    batch_results = os.path.join(out_folder, "batch.ibflowresults")
    logger.info(f"getting flow results from : {batch_results}")
    results = await sdk.get_flow_results(batch_results, options={"include_page_layouts": True})

    success = True
    for record in results["records"]:
        record_index = record["record_index"]
        if record_index != preds_dict["record_index"]:
            continue

        file_name = record["file_name"]

        page_layouts = record["layout"]["page_layouts"]
        page_numbers = [layout["page_number"] for layout in page_layouts]
        page_range = (min(page_numbers), max(page_numbers))
        train_pred = preds_dict["prediction"]
        train_page_range = (train_pred["page_start"], train_pred["page_end"])

        if page_range != train_page_range:
            logging.error(
                f"For file <{file_name}>, the page range from expected training prediction <{train_page_range}> "
                f"equal inference prediction <{page_range}> for record index: {record_index}"
            )
            success = False

        class_label = record["classification_label"]
        predicted_class_label = train_pred["annotated_class_name"]

        if class_label != predicted_class_label:
            logging.error(
                f"For file <{file_name}>, the class label from expected training prediction <{predicted_class_label}> "
                f"equal inference prediction <{class_label}>"
            )
            success = False

    logger.info(f"Inference test {'passed' if success else 'failed'}: {test_name}")

    await sdk.unpublish_solution(model_name, model_version)
    return success


async def run_inference_test(
    sdk: Instabase,
    wait_for: Optional[asyncio.Task],
    test_name: str,
    dataset_path: str,
    test_config: Dict,
) -> bool:
    logger = logging.getLogger(f"{sdk.name}: {test_name}")

    logger.info("Waiting for training test to finish before starting inference test")
    await wait_for
    success, job_id = wait_for.result()
    if not success:
        return False  # fail fast; if training failed, skip inference

    logger.info(f"Running inference test for training job {job_id}")

    model_project_path = Path(os.path.join(dataset_path, test_config["model_project_path"]))
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

    dataset_project_path = Path(os.path.join(dataset_path, test_config["dataset_project_path"]))
    dev_input_folder = dataset_project_path / "out_annotations" / "s1_process_files"

    logger.debug(
        f"Unload the model '{test_name}' in case it's already present. "
        "Repeating 10 times to capture multiple potential pods."
    )
    for _ in range(10):
        await sdk.unload_model(model_name, model_version)

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
        ibdoc_path = preds_dict["ibdoc_path"]
        preds_by_field = model_result_by_record[ibdoc_path + "-0"]
        field_name = prediction["field_name"]
        for annotation in prediction["annotations"]:
            train_val = annotation["value"]
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
    async with PUBLISH_LOCK:
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
    # Needed to add this since REMOTE_TEMP_ZIP_PATH would be overwritten
    # since we now have both ibformers_extraction and ibformers_classification
    remote_temp_zip_path = "%s.ibsolution" % uuid.uuid4().hex
    await sdk.delete_file(remote_code_loation, recursive=True)
    logger.debug(f"Writing to {remote_temp_zip_path}")
    await sdk.write_file(remote_temp_zip_path, contents)
    logger.debug(f"Unzipping {remote_temp_zip_path} to remote_code_loation")
    await sdk.unzip(zip_filepath=remote_temp_zip_path, destination=remote_code_loation)
    logger.debug(f"Deleting {remote_temp_zip_path}")
    await sdk.delete_file(remote_temp_zip_path)
    logger.debug(f"Done syncing code")


async def run_tests(train: bool, inference: bool, test_name: Optional[str], test_environment: str) -> None:
    model_tests = load_model_tests()
    if not model_tests:
        logging.error("No tests found in model_tests.yaml.")
        exit(1)

    if test_name:
        model_tests = {k: v for k, v in model_tests.items() if k == test_name}
        if not model_tests:
            logging.error(f"Test '{test_name}' not found in list of tests {list(model_tests.keys())}")
            exit(1)

    # validate test_environment
    envs = load_environments()
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
        root_path=os.path.join(env_config["path"], REMOTE_CODE_PREFIX),
    )

    remote_code_loation = {}
    if train:
        for package in [PackageType.EXTRACTION.value, PackageType.CLASSIFICATION.value]:
            # this solves race condition when run tests concurrently
            remote_code_loation[package] = "ibformers_%s" % uuid.uuid4().hex
            zip_bytes = zip_project(PROJECT_ROOT, package)
            sync_tasks[f"{test_environment}_{package}"] = asyncio.create_task(
                sync_and_unzip(sdk, zip_bytes, remote_code_loation[package])
            )
            del zip_bytes

    tasks: List[asyncio.Task] = []
    for test_name, test_config in model_tests.items():
        supported_envs = test_config["env"]
        package = test_config.get("package", PackageType.EXTRACTION.value)
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
            root_path=os.path.join(env_config["path"], REMOTE_CODE_PREFIX),
        )
        train_task = None
        if train:
            train_task = asyncio.create_task(
                run_training_test(
                    sdk,
                    wait_for=sync_tasks[f"{test_environment}_{package}"],
                    test_name=test_name,
                    root_path=env_config["path"],
                    remote_code_loation=remote_code_loation[package],
                    test_config=test_config,
                    package=package,
                )
            )
            tasks.append(train_task)
        if inference:
            if package == PackageType.EXTRACTION.value:
                inference_task = asyncio.create_task(
                    run_inference_test(
                        sdk,
                        wait_for=train_task,
                        test_name=test_name,
                        dataset_path=env_config["path"],
                        test_config=test_config,
                    )
                )
                tasks.append(inference_task)
            else:
                inference_task = asyncio.create_task(
                    run_inference_test_classifier(
                        sdk,
                        wait_for=train_task,
                        test_name=test_name,
                        dataset_path=env_config["path"],
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
