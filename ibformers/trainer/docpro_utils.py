import logging
import os
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

from transformers import HfArgumentParser

from ibformers.data.collators.augmenters.args import AugmenterArguments
from ibformers.trainer.arguments import (
    ModelArguments,
    DataAndPipelineArguments,
    IbArguments,
    EnhancedTrainingArguments,
    ExtraModelArguments,
)
from ibformers.trainer.ib_utils import (
    MountDetails,
    prepare_ib_params,
    InstabaseSDK,
    _abspath,
)
from ibformers.trainer.train import run_train
from ibformers.trainer.arguments import ModelArguments, DataAndPipelineArguments, IbArguments, EnhancedTrainingArguments
from ibformers.callbacks.split_classifier import SplitClassifierCallback
from ibformers.callbacks.extraction import DocProCallback
from ibformers.callbacks.classifier import DocProClassificationCallback
from instabase.dataset_utils.sdk import LocalDatasetSDK, RemoteDatasetSDK
from instabase.training_utils.model_artifact import ModelArtifactTemplateGenerator
from ibformers.utils.print_dir import print_dir


def prepare_classification_params(
    hyperparams: Dict,
    dataset_list: List,
    save_path: str,
    final_model_dir: str,
    file_client: Any,
    username: str,
    job_metadata_client: Any,
    mount_details: Optional[MountDetails] = None,
    model_name: str = "CustomModel",
):
    """
    Handles defaults for doc-pro and set up special parameters
    :param hyperparams: dictionary of hyperparams passed from the frontend
    :param dataset_list: list of paths, can be either local or remote
    :param save_path: ib location of the training job output
    :param final_model_dir: where to save model files on the local fs
    :param file_client: file_client used to open remote files
    :param username: username who run the training job
    :param job_metadata_client: client used by callback to log progress/status of the training
    :param mount_details: optional details of s3 mount
    :param model_name: name of the model used in front end
    :return:
    """

    out_dict = prepare_ib_params(
        hyperparams,
        None,
        save_path,
        file_client,
        username,
        job_metadata_client,
        mount_details,
        model_name,
    )

    out_dict["dataset_name_or_path"] = "ib_classification"
    out_dict["dataset_config_name"] = "ib_classification"
    out_dict["train_file"] = dataset_list
    out_dict["final_model_dir"] = final_model_dir
    out_dict["report_to"] = "none"
    out_dict["disable_tqdm"] = True

    return out_dict


def prepare_split_classification_params(
    hyperparams: Dict,
    dataset_list: List,
    save_path: str,
    final_model_dir: str,
    file_client: Any,
    username: str,
    job_metadata_client: Any,
    mount_details: Optional[MountDetails] = None,
    model_name: str = "CustomModel",
):
    """
    Handles defaults for doc-pro and set up special parameters
    :param hyperparams: dictionary of hyperparams passed from the frontend
    :param dataset_list: list of paths, can be either local or remote
    :param save_path: ib location of the training job output
    :param final_model_dir: where to save model files on the local fs
    :param file_client: file_client used to open remote files
    :param username: username who run the training job
    :param job_metadata_client: client used by callback to log progress/status of the training
    :param mount_details: optional details of s3 mount
    :param model_name: name of the model used in front end
    :return:
    """

    out_dict = prepare_ib_params(
        hyperparams,
        None,
        save_path,
        file_client,
        username,
        job_metadata_client,
        mount_details,
        model_name,
    )

    out_dict["dataset_name_or_path"] = "ib_split_class"
    out_dict["dataset_config_name"] = "ib_split_class"
    out_dict["label_names"] = ["sc_labels"]
    out_dict["train_file"] = dataset_list
    out_dict["final_model_dir"] = final_model_dir
    out_dict["report_to"] = "none"
    out_dict["disable_tqdm"] = True

    return out_dict


def prepare_docpro_params(
    hyperparams: Dict,
    dataset_list: List,
    save_path: str,
    final_model_dir: str,
    extraction_class_name: str,
    file_client: Any,
    username: str,
    job_metadata_client: Any,
    mount_details: Optional[MountDetails] = None,
    model_name: str = "CustomModel",
):
    """
    Handles defaults for doc-pro and set up special parameters
    :param hyperparams: dictionary of hyperparams passed from the frontend
    :param dataset_list: list of paths, can be either local or remote
    :param save_path: ib location of the training job output
    :param final_model_dir: where to save model files on the local fs
    :param extraction_class_name: name of the extracted class
    :param file_client: file_client used to open remote files
    :param username: username who run the training job
    :param job_metadata_client: client used by callback to log progress/status of the training
    :param mount_details: optional details of s3 mount
    :param model_name: name of the model used in front end
    :return:
    """

    out_dict = prepare_ib_params(
        hyperparams,
        None,
        save_path,
        file_client,
        username,
        job_metadata_client,
        mount_details,
        model_name,
    )

    out_dict["dataset_name_or_path"] = hyperparams.get("dataset_name_or_path", "ib_extraction")
    out_dict["dataset_config_name"] = hyperparams.get("dataset_config_name", "ib_extraction")
    out_dict["train_file"] = dataset_list
    out_dict["extraction_class_name"] = extraction_class_name
    out_dict["final_model_dir"] = final_model_dir
    out_dict["report_to"] = "none"
    out_dict["disable_tqdm"] = True
    out_dict["max_train_samples"] = hyperparams.get("max_train_samples", None)
    out_dict["label_names"] = hyperparams.get("label_names", None)
    out_dict["pad_to_max_length"] = hyperparams.get("pad_to_max_length", False)

    return out_dict


def load_datasets(dataset_paths, ibsdk):
    assert isinstance(dataset_paths, list)

    file_client = ibsdk.file_client
    username = ibsdk.username
    try:
        # load from doc pro
        if file_client is None:
            datasets_list = [LocalDatasetSDK(dataset_path) for dataset_path in dataset_paths]
        else:
            datasets_list = [RemoteDatasetSDK(dataset_path, file_client, username) for dataset_path in dataset_paths]

    except Exception as e:
        logging.error(traceback.format_exc())
        raise RuntimeError(f"Error while compiling the datasets: {e}") from e

    return datasets_list


def run_train_both_classification(hyperparams: Dict, **kwargs):
    task_type = hyperparams.get("task_type")
    if task_type is None:
        raise ValueError("Need to provide task type parameter for classification")
    elif task_type == "split_classification":
        return run_train_split_classification(hyperparams, **kwargs)
    elif task_type == "classification":
        return run_train_classification(hyperparams, **kwargs)
    else:
        raise ValueError("task_type can be either split_classification or classification")


def run_train_split_classification(
    hyperparams: Dict,
    dataset_paths: List[str],
    save_path: str,
    file_client: Any,
    username: str,
    job_metadata_client: Any,
    mount_details: Optional[MountDetails] = None,
    model_name: str = "CustomModel",
    **kwargs: Any,
):
    """
    Endpoint used to run doc pro jobs.

    :param hyperparams: dictionary of hyperparams passed from the frontend
    :param dataset_paths: list of paths, can be either local or remote
    :param save_path: ib location of the training job output
    :param file_client: file_client used to open remote files
    :param username: username of user who runs the training job
    :param job_metadata_client: client used by callback to log progress/status of the training
    :param mount_details: optional details of s3 mount
    :param model_name: name of the model used in front end
    :param kwargs:
    :return:
    """
    logging.info("Starting Doc Pro Split Classification Model Training ----------")
    logging.info("Arguments to this training session:")
    logging.info(f"Hyperparameters: {hyperparams}")
    logging.info(f"Dataset Paths: {dataset_paths}")
    logging.info(f"Model Name: {model_name}")
    logging.info(f"Save Path: {save_path}")

    # Generate local folder to save in
    logging.info("Creating Model Service Model template...")

    template_path = _abspath(f"ib_package/ModelServiceTemplate/split_classification")
    if not Path(template_path).is_dir():
        logging.error(f"Directory with template files ({template_path}) does not exist")

    context = ModelArtifactTemplateGenerator(
        file_client,
        username,
        template_path,
        model_name,
        model_name,
        model_name,
        {"training_job_id": job_metadata_client.job_id},
    ).generate()
    save_folder = context.tmp_dir.name
    save_model_dir = os.path.join(context.artifact_path, f"src/py/{model_name}/model_data")

    # Debug folder structure
    logging.info("Copied Model Service Model template to local file system")
    logging.info("The folder structure so far is:")
    print_dir(save_folder)

    assert hyperparams is not None
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataAndPipelineArguments,
            EnhancedTrainingArguments,
            IbArguments,
            AugmenterArguments,
        )
    )

    if hasattr(file_client, "file_client") and file_client.file_client is None:
        # support for InstabaseSDKDummy - debugging only
        ibsdk = file_client
    else:
        ibsdk = InstabaseSDK(file_client, username)

    dataset_list = load_datasets(dataset_paths, ibsdk)

    hparams_dict = prepare_split_classification_params(
        hyperparams,
        dataset_paths,
        save_path,
        save_model_dir,
        file_client,
        username,
        job_metadata_client,
        mount_details,
        model_name,
    )

    if hyperparams.get("debug_cuda_launch_blocking", False):
        logging.warning("Setting up debbuging mode (CUDA_LAUNCH_BLOCKING=1)")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    model_args, data_args, training_args, ib_args, augmenter_args = parser.parse_dict(hparams_dict)

    callback = SplitClassifierCallback(
        dataset_list=dataset_list,
        artifacts_context=context,
        job_metadata_client=ib_args.job_metadata_client,
        ibsdk=ibsdk,
        username=ib_args.username,
        mount_details=ib_args.mount_details,
        model_name=ib_args.model_name,
        ib_save_path=ib_args.ib_save_path,
    )

    run_train(
        model_args,
        data_args,
        training_args,
        ib_args,
        augmenter_args,
        ExtraModelArguments(None),
        extra_callbacks=[callback],
        extra_load_kwargs={"ibsdk": ibsdk},
    )

    return {"results": "Finished"}


def run_train_classification(
    hyperparams: Dict,
    dataset_paths: List[str],
    save_path: str,
    file_client: Any,
    username: str,
    job_metadata_client: Any,
    mount_details: Optional[MountDetails] = None,
    model_name: str = "CustomModel",
    **kwargs: Any,
):
    """
    Endpoint used to run doc pro jobs.

    :param hyperparams: dictionary of hyperparams passed from the frontend
    :param dataset_paths: list of paths, can be either local or remote
    :param save_path: ib location of the training job output
    :param file_client: file_client used to open remote files
    :param username: username of user who runs the training job
    :param job_metadata_client: client used by callback to log progress/status of the training
    :param mount_details: optional details of s3 mount
    :param model_name: name of the model used in front end
    :param kwargs:
    :return:
    """
    logging.info("Starting Doc Pro Classification Model Training ----------")
    logging.info("Arguments to this training session:")
    logging.info(f"Hyperparameters: {hyperparams}")
    logging.info(f"Dataset Paths: {dataset_paths}")
    logging.info(f"Model Name: {model_name}")
    logging.info(f"Save Path: {save_path}")

    # Generate local folder to save in
    logging.info("Creating Model Service Model template...")

    template_path = _abspath(f"ib_package/ModelServiceTemplate/classification")
    if not Path(template_path).is_dir():
        logging.error(f"Directory with template files ({template_path}) does not exist")

    context = ModelArtifactTemplateGenerator(
        file_client,
        username,
        template_path,
        model_name,
        model_name,
        model_name,
        {"training_job_id": job_metadata_client.job_id},
    ).generate()
    save_folder = context.tmp_dir.name
    save_model_dir = os.path.join(context.artifact_path, f"src/py/{model_name}/model_data")

    # Debug folder structure
    logging.info("Copied Model Service Model template to local file system")
    logging.info("The folder structure so far is:")
    print_dir(save_folder)

    assert hyperparams is not None
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataAndPipelineArguments,
            EnhancedTrainingArguments,
            IbArguments,
            AugmenterArguments,
        )
    )

    if hasattr(file_client, "file_client") and file_client.file_client is None:
        # support for InstabaseSDKDummy - debugging only
        ibsdk = file_client
    else:
        ibsdk = InstabaseSDK(file_client, username)

    dataset_list = load_datasets(dataset_paths, ibsdk)

    hparams_dict = prepare_classification_params(
        hyperparams,
        dataset_paths,
        save_path,
        save_model_dir,
        file_client,
        username,
        job_metadata_client,
        mount_details,
        model_name,
    )

    if hyperparams.get("debug_cuda_launch_blocking", False):
        logging.warning("Setting up debbuging mode (CUDA_LAUNCH_BLOCKING=1)")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    model_args, data_args, training_args, ib_args, augmenter_args = parser.parse_dict(hparams_dict)

    callback = DocProClassificationCallback(
        dataset_list=dataset_list,
        artifacts_context=context,
        job_metadata_client=ib_args.job_metadata_client,
        ibsdk=ibsdk,
        username=ib_args.username,
        mount_details=ib_args.mount_details,
        model_name=ib_args.model_name,
        ib_save_path=ib_args.ib_save_path,
    )

    run_train(
        model_args,
        data_args,
        training_args,
        ib_args,
        augmenter_args,
        ExtraModelArguments(None),
        extra_callbacks=[callback],
        extra_load_kwargs={"ibsdk": ibsdk},
    )

    return {"results": "Finished"}


def run_train_doc_pro(
    hyperparams: Dict,
    dataset_paths: List[str],
    save_path: str,
    extraction_class_name: str,
    file_client: Any,
    username: str,
    job_metadata_client: Any,
    mount_details: Optional[MountDetails] = None,
    model_name: str = "CustomModel",
    **kwargs: Any,
):
    """
    Endpoint used to run doc pro jobs.

    :param hyperparams: dictionary of hyperparams passed from the frontend
    :param dataset_paths: list of paths, can be either local or remote
    :param save_path: ib location of the training job output
    :param extraction_class_name: name of the extracted class
    :param file_client: file_client used to open remote files
    :param username: username of user who runs the training job
    :param job_metadata_client: client used by callback to log progress/status of the training
    :param mount_details: optional details of s3 mount
    :param model_name: name of the model used in front end
    :param kwargs:
    :return:
    """
    logging.info("Starting Doc Pro Extraction Model Training ----------")
    logging.info("Arguments to this training session:")
    logging.info(f"Hyperparameters: {hyperparams}")
    logging.info(f"Dataset Paths: {dataset_paths}")
    logging.info(f"Model Name: {model_name}")
    logging.info(f"Save Path: {save_path}")
    logging.info(f"Extraction Class Name: {extraction_class_name}")

    # Generate local folder to save in
    logging.info("Creating Model Service Model template...")
    template_path = _abspath(f"ib_package/ModelServiceTemplate/extraction")
    if not Path(template_path).is_dir():
        logging.error(f"Directory with template files ({template_path}) does not exist")

    context = ModelArtifactTemplateGenerator(
        file_client,
        username,
        template_path,
        model_name,
        model_name,
        model_name,
        {"training_job_id": job_metadata_client.job_id},
    ).generate()
    save_folder = context.tmp_dir.name
    save_model_dir = os.path.join(context.artifact_path, f"src/py/{model_name}/model_data")

    # Debug folder structure
    logging.info("Copied Model Service Model template to local file system")
    logging.info("The folder structure so far is:")
    print_dir(save_folder)

    assert hyperparams is not None
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataAndPipelineArguments,
            EnhancedTrainingArguments,
            IbArguments,
            AugmenterArguments,
            ExtraModelArguments,
        )
    )

    if hasattr(file_client, "file_client") and file_client.file_client is None:
        # support for InstabaseSDKDummy - debugging only
        ibsdk = file_client
    else:
        ibsdk = InstabaseSDK(file_client, username)

    dataset_list = load_datasets(dataset_paths, ibsdk)

    hparams_dict = prepare_docpro_params(
        hyperparams,
        dataset_paths,
        save_path,
        save_model_dir,
        extraction_class_name,
        file_client,
        username,
        job_metadata_client,
        mount_details,
        model_name,
    )

    if hyperparams.get("debug_cuda_launch_blocking", False):
        logging.warning("Setting up debbuging mode (CUDA_LAUNCH_BLOCKING=1)")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    model_args, data_args, training_args, ib_args, augmenter_args, extra_model_args = parser.parse_dict(hparams_dict)

    extra_load_kwargs = {"ibsdk": ibsdk, "extraction_class_name": extraction_class_name}

    callback = DocProCallback(
        dataset_list=dataset_list,
        artifacts_context=context,
        extraction_class_name=extraction_class_name,
        job_metadata_client=ib_args.job_metadata_client,
        ibsdk=ibsdk,
        username=ib_args.username,
        mount_details=ib_args.mount_details,
        model_name=ib_args.model_name,
        ib_save_path=ib_args.ib_save_path,
        log_metrics_to_metadata=hyperparams.get("log_metrics_to_metadata", False),
    )

    run_train(
        model_args,
        data_args,
        training_args,
        ib_args,
        augmenter_args,
        extra_model_args,
        extra_callbacks=[callback],
        extra_load_kwargs=extra_load_kwargs,
    )

    return {"results": "Finished"}
