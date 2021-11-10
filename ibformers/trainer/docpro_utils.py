import json
import logging
import os
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

from transformers import HfArgumentParser, TrainingArguments, TrainerCallback

from ibformers.trainer.ib_utils import (
    MountDetails,
    prepare_ib_params,
    InstabaseSDK,
    _abspath,
    upload_dir,
)
from ibformers.trainer.refiner_module_generator import write_refiner_program
from ibformers.trainer.train import run_train
from ibformers.trainer.train_utils import ModelArguments, DataAndPipelineArguments, IbArguments
from instabase.dataset_utils.sdk import LocalDatasetSDK, RemoteDatasetSDK, DatasetSDK
from instabase.dataset_utils.shared_types import (
    PredictionResultDict,
    PredictionInstanceDict,
    PredictionFieldDict,
    IndexedWordDict,
)
from instabase.training_utils.model_artifact import (
    ModelArtifactTemplateGenerator,
    ModelArtifactContext,
    ModelArtifactMetricsWriter,
    TableMetric,
    ValueMetric,
    ModelArtifactPredictionsWriter,
)


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
    model_name: str = 'CustomModel',
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

    out_dict["dataset_name_or_path"] = "docpro_ds"
    out_dict["dataset_config_name"] = "docpro_ds"
    out_dict["train_file"] = dataset_list
    out_dict["extraction_class_name"] = extraction_class_name
    out_dict["final_model_dir"] = final_model_dir

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
            datasets_list = [
                RemoteDatasetSDK(dataset_path, file_client, username)
                for dataset_path in dataset_paths
            ]

    except Exception as e:
        logging.error(traceback.format_exc())
        raise RuntimeError(f"Error while compiling the datasets: {e}") from e

    return datasets_list


class DocProCallback(TrainerCallback):
    """
    Handles events specific to doc pro
    """

    # list of files/dirs which will not be copied into the package
    ibformers_do_not_copy = ['hf_token.py']

    def __init__(
        self,
        dataset_list: List[DatasetSDK],
        artifacts_context: ModelArtifactContext,
        extraction_class_name: str,
        job_metadata_client: "JobMetadataClient",
        ibsdk: InstabaseSDK,
        username: str,
        mount_details: Dict,
        model_name: str,
        ib_save_path: str,
    ):
        """
        :param dataset_list: List of dataset sdk objects
        :param artifacts_context: ModelArtifactContext which handles creation of model artifacts
        :param extraction_class_name: name of the extracted class
        :param job_metadata_client: client used by callback to log progress/status of the training
        :param ibsdk: sdk used to transfer files to IB fs
        :param username: username of user who run the training job
        :param mount_details: optional details of s3 mount
        :param model_name: name of the model used in front end
        :param ib_save_path: ib location of the training job output

        """
        self.extraction_class_name = extraction_class_name
        self.dataset_list = dataset_list
        self.artifacts_context = artifacts_context
        self.ib_save_path = ib_save_path
        self.model_name = model_name
        self.mount_details = mount_details
        self.username = username
        self.job_metadata_client = job_metadata_client
        self.evaluation_results = None
        self.prediction_results = None
        self.ibsdk = ibsdk
        self.job_status = {}
        self.metrics_writer = ModelArtifactMetricsWriter(artifacts_context)
        self.prediction_writer = ModelArtifactPredictionsWriter(artifacts_context)
        self.save_folder = artifacts_context.tmp_dir.name
        self.save_model_dir = os.path.join(
            artifacts_context.artifact_path, f'src/py/{model_name}/model_data'
        )
        self.id_to_dataset = {dataset.metadata['id']: dataset for dataset in dataset_list}

    def copy_library_src_to_package(self):
        # copy ibformers lib into the package

        ibformers_path = Path(_abspath('')).parent
        assert (
            ibformers_path.name == 'ibformers'
        ), f'ibformers_path is wrong. Path: {ibformers_path}'
        # TODO(rafal): once ibformers is converted to relative imports copy ibformers into py/package_name/ibformers dir
        # py directory location
        py_directory = Path(self.artifacts_context.artifact_path) / 'src' / 'py'
        shutil.copytree(
            ibformers_path,
            py_directory / 'ibformers',
            ignore=lambda x, y: self.ibformers_do_not_copy,
        )

    def move_data_to_ib(self):
        self.copy_library_src_to_package()

        logging.info("Final state of the Model Artifact folder structure:")
        _print_dir(self.save_folder)

        logging.info("Uploading")
        self.job_metadata_client.update_message('UPLOADING MODEL')
        upload_dir(
            sdk=self.ibsdk,
            local_folder=self.save_folder,
            remote_folder=self.ib_save_path,
            mount_details=self.mount_details,
        )
        logging.info("Uploaded")

    def set_status(self, new_status: Dict):
        self.job_status.update(new_status)
        self.job_metadata_client.update_metadata(self.job_status)

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.set_status({"progress": state.global_step / state.max_steps})

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.set_status({"progress": state.global_step / state.max_steps})

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            # workaround for missing on_predict callback in the transformers TrainerCallback
            if 'predict_loss' in kwargs["metrics"]:
                self.on_predict(args, state, control, **kwargs)
            else:
                metrics = {}
                metrics['precision'] = kwargs["metrics"]['eval_precision']
                metrics['recall'] = kwargs["metrics"]['eval_recall']
                metrics['f1'] = kwargs["metrics"]['eval_f1']
                self.set_status(
                    {"evaluation_results": metrics, "progress": state.global_step / state.max_steps}
                )

                self.evaluation_results = metrics

    def write_metrics(self):
        logging.info("Writing metrics for this training run...")
        metrics_writer = self.metrics_writer
        overall_accuracy = "Unknown"
        try:
            evaluation_metrics = self.job_status.get('evaluation_results')
            if evaluation_metrics:

                # First add the table
                headers = list(evaluation_metrics.keys())
                all_rows = set()
                for column in evaluation_metrics:
                    all_rows.update(list(evaluation_metrics[column].keys()))
                sorted_rows = list(all_rows)
                sorted_rows.sort()

                rows = []
                for row_key in sorted_rows:
                    rows.append(
                        [row_key, *[evaluation_metrics[c].get(row_key, '') for c in headers]]
                    )

                metrics_writer.add_table_metric(
                    TableMetric(
                        title='Field-level Metrics',
                        subtitle='Accuracy scores for individual fields learned by the model',
                        headers=["Field Name"] + headers,
                        rows=rows,
                    )
                )

                # Then add high-level metrics
                f1_scores = [x for x in evaluation_metrics['f1'].values() if x != 'NAN']
                avg_f1 = (
                    "{:.3f}".format(sum(f1_scores) / float(len(f1_scores))) if f1_scores else 'N/A'
                )
                overall_accuracy = (
                    "{:.2f}%".format(sum(f1_scores) * 100.0 / float(len(f1_scores)))
                    if f1_scores
                    else 'Unknown'
                )
                metrics_writer.add_high_level_metric(
                    ValueMetric(
                        title='Macro F1',
                        subtitle='Average F1 score across all fields',
                        value=avg_f1,
                        tag_text='ACCURACY',
                        tag_type='INFO',
                    )
                )
                precisions = [x for x in evaluation_metrics['precision'].values() if x != 'NAN']
                avg_precision = (
                    "{:.2f}%".format(sum(precisions) * 100.0 / float(len(precisions)))
                    if precisions
                    else 'N/A'
                )
                metrics_writer.add_high_level_metric(
                    ValueMetric(
                        title='Macro Precision',
                        subtitle='The average precision across all fields',
                        value=avg_precision,
                        tag_text='ACCURACY',
                        tag_type='INFO',
                    )
                )
                recalls = [x for x in evaluation_metrics['recall'].values() if x != 'NAN']
                avg_recall = (
                    "{:.2f}%".format(sum(recalls) * 100.0 / float(len(recalls)))
                    if recalls
                    else 'N/A'
                )
                metrics_writer.add_high_level_metric(
                    ValueMetric(
                        title='Average Recall',
                        subtitle='The average recall across all fields',
                        value=avg_recall,
                        tag_text='ACCURACY',
                        tag_type='INFO',
                    )
                )
        except Exception as e:
            logging.error(traceback.format_exc())

        try:
            logging.info("Writing metrics for this training run...")
            metrics_writer.write()
        except Exception as e:
            logging.error("Could not write metrics")
            logging.error(traceback.format_exc())

        # Set the overall accuracy of the model
        self.set_status({'accuracy': overall_accuracy})

    def write_predictions(self, predictions_dict):
        # Now write the predictions
        prediction_writer = self.prediction_writer

        logging.info("Writing prediction results to IB")
        try:
            for record_name, record_value in predictions_dict.items():
                # TODO(VONTELL) - rn we are using the field name directly.
                # Using the vars below, we could get the actual ids...
                dataset_id = record_name[:36]
                dataset = self.id_to_dataset[dataset_id]
                # This is because the first 37 chars [0]-[36] are reserved for the UUID (36), and a dash (-)
                ibdoc_filename = record_name[37 : record_name.index('.ibdoc') + len('.ibdoc')]
                # This grabs the record index at the end of the file name
                record_idx = int(
                    record_name[len(dataset_id) + len(ibdoc_filename) + 2 : -1 * len('.json')]
                )
                logging.info(f'Saving prediction for {dataset_id}, {ibdoc_filename}, {record_idx}')

                # Currently, our models only outputs a single entity
                # Predictions, however, can support multiple
                fields = []
                is_test_file = record_value['is_test_file']
                record_entities = record_value['entities']
                for field_name, field_value in record_entities.items():
                    indexed_words = [
                        IndexedWordDict(
                            line_index=w['word_line_idx'], word_index=w['word_in_line_idx']
                        )
                        for w in field_value['words']
                    ]
                    predictions = (
                        [
                            PredictionInstanceDict(
                                avg_confidence=field_value['avg_confidence'],
                                value=field_value['text'],
                                words=indexed_words,
                            )
                        ]
                        if len(indexed_words) > 0
                        else []
                    )

                    fields.append(
                        PredictionFieldDict(field_name=field_name, annotations=predictions)
                    )
                prediction_writer.add_prediction(
                    dataset,
                    ibdoc_filename,
                    record_idx,
                    PredictionResultDict(
                        annotated_class_name=self.extraction_class_name, fields=fields
                    ),
                    is_test_file,
                )
        except Exception as e:
            logging.error(traceback.format_exc())
        try:
            logging.info("Writing predictions for this training run...")
            prediction_writer.write()
        except Exception as e:
            logging.error("Could not write prediction")
            logging.error(traceback.format_exc())
        self.set_status({'predictions_uuid': uuid.uuid4().hex})

    def generate_refiner(self, label_names):
        logging.info("Generating the Refiner module for this model...")

        try:
            ib_model_path = os.path.join(self.ib_save_path, 'artifact')
            dev_path = os.path.join(
                self.dataset_list[0].dataset_path, self.dataset_list[0].metadata['docs_path']
            )
            write_refiner_program(
                self.artifacts_context, ib_model_path, label_names, self.model_name, dev_path
            )
            logging.info("Finished generating the Refiner module")
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Skipped Refiner module generation due to an error: {e}")

    def on_predict(self, args, state, control, **kwargs):
        # called after the training finish
        predictions = kwargs["metrics"]['predict_predictions']
        # FINALIZE STEPS
        self.write_metrics()
        self.write_predictions(predictions)

        id2label = kwargs["model"].config.id2label
        if id2label[0] != "O":
            logging.error(f'0 index for label should be asigned to O class. Got: {id2label[0]}')
        label_names = [id2label[idx] for idx in range(1, len(id2label))]
        self.generate_refiner(label_names)
        self.move_data_to_ib()


def _print_dir(path):
    for dirpath, dirnames, filenames in os.walk(path):
        directory_level = dirpath.replace(path, "")
        directory_level = directory_level.count(os.sep)
        indent = " " * 4
        logging.info("{}{}/".format(indent * directory_level, os.path.basename(dirpath)))

        for f in filenames:
            logging.info("{}{}".format(indent * (directory_level + 1), f))


def run_train_doc_pro(
    hyperparams: Dict,
    dataset_paths: List[str],
    save_path: str,
    extraction_class_name: str,
    file_client: Any,
    username: str,
    job_metadata_client: Any,
    mount_details: Optional[MountDetails] = None,
    model_name: str = 'CustomModel',
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
    logging.info('Starting Doc Pro Extraction Model Training ----------')
    logging.info('Arguments to this training session:')
    logging.info(f'Hyperparameters: {hyperparams}')
    logging.info(f'Dataset Paths: {dataset_paths}')
    logging.info(f'Model Name: {model_name}')
    logging.info(f'Save Path: {save_path}')
    logging.info(f'Extraction Class Name: {extraction_class_name}')

    # Generate local folder to save in
    logging.info("Creating Model Service Model template...")
    template_path = _abspath('ib_package/ModelServiceTemplate')
    if not Path(template_path).is_dir():
        logging.error(f"Directory with template files ({template_path}) does not exist")

    context = ModelArtifactTemplateGenerator(
        file_client,
        username,
        template_path,
        model_name,
        model_name,
        model_name,
        {'training_job_id': job_metadata_client.job_id},
    ).generate()
    save_folder = context.tmp_dir.name
    save_model_dir = os.path.join(context.artifact_path, f'src/py/{model_name}/model_data')

    # Debug folder structure
    logging.info("Copied Model Service Model template to local file system")
    logging.info("The folder structure so far is:")
    _print_dir(save_folder)

    assert hyperparams is not None
    parser = HfArgumentParser(
        (ModelArguments, DataAndPipelineArguments, TrainingArguments, IbArguments)
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

    model_args, data_args, training_args, ib_args = parser.parse_dict(hparams_dict)

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
    )

    run_train(
        model_args,
        data_args,
        training_args,
        ib_args,
        extra_callbacks=[callback],
        extra_load_kwargs={"ibsdk": ibsdk, "extraction_class_name": extraction_class_name},
    )

    return {'results': 'Finished'}