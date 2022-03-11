from copy import deepcopy
from transformers import TrainerCallback
from typing import Dict, List, Any, Optional
import traceback
import logging
import os
import shutil
from pathlib import Path
from instabase.dataset_utils.sdk import LocalDatasetSDK, RemoteDatasetSDK, DatasetSDK
from ibformers.trainer.ib_utils import (
    MountDetails,
    prepare_ib_params,
    InstabaseSDK,
    _abspath,
    upload_dir,
)
from ibformers.trainer.splitclassifier_module_generator import write_classifier_module
from instabase.training_utils.model_artifact import (
    ModelArtifactTemplateGenerator,
    ModelArtifactContext,
    ModelArtifactMetricsWriter,
    TableMetric,
    ValueMetric,
    ModelArtifactPredictionsWriter,
)
from instabase.dataset_utils.shared_types import (
    PredictionResultDict,
    PredictionInstanceDict,
    PredictionFieldDict,
    IndexedWordDict,
)
import pandas as pd
import uuid


class SplitClassifierCallback(TrainerCallback):
    """
    Handles events specific to doc pro
    """

    # list of files/dirs which will not be copied into the package
    ibformers_do_not_copy = ["hf_token.py"]

    def __init__(
        self,
        dataset_list: List[DatasetSDK],
        artifacts_context: ModelArtifactContext,
        job_metadata_client: "JobMetadataClient",  # type: ignore
        ibsdk: InstabaseSDK,
        username: str,
        mount_details: Dict,
        model_name: str,
        ib_save_path: str,
    ):
        """
        :param dataset_list: List of dataset sdk objects
        :param artifacts_context: ModelArtifactContext which handles creation of model artifacts
        :param job_metadata_client: client used by callback to log progress/status of the training
        :param ibsdk: sdk used to transfer files to IB fs
        :param username: username of user who run the training job
        :param mount_details: optional details of s3 mount
        :param model_name: name of the model used in front end
        :param ib_save_path: ib location of the training job output

        """
        self.dataset_list = dataset_list
        self.artifacts_context = artifacts_context
        self.ib_save_path = ib_save_path
        self.model_name = model_name
        self.mount_details = mount_details
        self.username = username
        self.job_metadata_client = job_metadata_client
        self.evaluation_results = []
        self.prediction_results = None
        self.ibsdk = ibsdk
        self.job_status = {}
        self.metrics_writer = ModelArtifactMetricsWriter(artifacts_context)
        self.prediction_writer = ModelArtifactPredictionsWriter(artifacts_context)
        self.save_folder = artifacts_context.tmp_dir.name
        self.save_model_dir = os.path.join(artifacts_context.artifact_path, f"src/py/{model_name}/model_data")
        self.id_to_dataset = {dataset.metadata["id"]: dataset for dataset in dataset_list}

    def copy_library_src_to_package(self):
        # copy ibformers lib into the package

        ibformers_path = Path(_abspath("")).parent
        assert ibformers_path.name == "ibformers", f"ibformers_path is wrong. Path: {ibformers_path}"
        # TODO(rafal): once ibformers is converted to relative imports copy ibformers into py/package_name/ibformers dir
        # py directory location
        py_directory = Path(self.artifacts_context.artifact_path) / "src" / "py"
        shutil.copytree(
            ibformers_path,
            py_directory / "ibformers",
            ignore=lambda x, y: self.ibformers_do_not_copy,
        )

    def move_data_to_ib(self):
        self.copy_library_src_to_package()

        logging.info("Final state of the Model Artifact folder structure:")

        logging.info("Uploading")
        self.job_metadata_client.update_message("UPLOADING MODEL")
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
            if "predict_loss" in kwargs["metrics"]:
                self.on_predict(args, state, control, **kwargs)
            elif "eval_loss" in kwargs["metrics"]:
                metrics = {}
                splitter_metrics = kwargs["metrics"]["eval_splitter_metrics"]
                classifier_metrics = kwargs["metrics"]["eval_classifier_metrics"]
                metrics["splitter_metrics"] = splitter_metrics
                metrics["classifier_metrics"] = classifier_metrics

                # Had to duplicate this so that CI tests work, since it currently supports only
                # two levels of nesting
                metrics.update({f"splitter_{k}": v for k, v in splitter_metrics.items()})
                metrics.update({f"classifier_{k}": v for k, v in classifier_metrics.items()})
                self.set_status({"evaluation_results": metrics, "progress": state.global_step / state.max_steps})

                self.evaluation_results.append(metrics)
            else:
                # ignore last evaluation call
                pass

    def write_metrics(self):
        logging.info("Writing metrics for this training run...")
        metrics_writer = self.metrics_writer
        try:
            evaluation_metrics = self.job_status.get("evaluation_results")
            if evaluation_metrics:

                splitter_metrics = evaluation_metrics["splitter_metrics"]
                splitter_accuracy = splitter_metrics["accuracy"]

                classifier_metrics = evaluation_metrics["classifier_metrics"]
                classifier_accuracy = classifier_metrics["accuracy"]

                metrics_writer.add_high_level_metric(
                    ValueMetric(
                        title="Accuracy for classifier",
                        subtitle="Accuracy",
                        value=f"{classifier_accuracy:.2f}",
                        tag_text="ACCURACY",
                        tag_type="INFO",
                    )
                )

                metrics_writer.add_high_level_metric(
                    ValueMetric(
                        title="Accuracy for splitter",
                        subtitle="Accuracy",
                        value=f"{splitter_accuracy:.2f}",
                        tag_text="ACCURACY",
                        tag_type="INFO",
                    )
                )
                splitter_table = pd.DataFrame(
                    {k: v for k, v in splitter_metrics.items() if k != "accuracy"}
                ).to_records(index=True)
                splitter_headers = list(splitter_table.dtype.names)
                splitter_headers[0] = "Class Type"
                splitter_rows = splitter_table.tolist()

                metrics_writer.add_table_metric(
                    TableMetric(
                        title="Splitter Metrics",
                        subtitle="Accuracy scores for split/no-split learned by the model",
                        headers=splitter_headers,
                        rows=splitter_rows,
                    )
                )

                classifier_table = pd.DataFrame(
                    {k: v for k, v in classifier_metrics.items() if k != "accuracy"}
                ).to_records(index=True)
                classifier_headers = list(classifier_table.dtype.names)
                classifier_headers[0] = "Class Type"
                classifier_rows = classifier_table.tolist()

                metrics_writer.add_table_metric(
                    TableMetric(
                        title="Classifier Metrics",
                        subtitle="Accuracy scores for classes learned by the model",
                        headers=classifier_headers,
                        rows=classifier_rows,
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
        overall_accuracy = f"{(splitter_accuracy + classifier_accuracy) / 2.0 * 100.0:.2f}%"
        self.set_status(
            {
                "splitter_accuracy": splitter_accuracy,
                "classifier_accuracy": classifier_accuracy,
                "accuracy": overall_accuracy,
            }
        )

    def write_predictions(self, predictions_dict):
        # Now write the predictions
        prediction_writer = self.prediction_writer

        logging.info("Writing prediction results to IB")
        try:
            for doc_id, prediction in predictions_dict.items():
                dataset_id = doc_id[:36]
                dataset = self.id_to_dataset[dataset_id]
                # This is because the first 37 chars [0]-[36] are reserved for the UUID (36), and a dash (-)
                ibdoc_filename = doc_id[37:]
                split_preds = []
                for pred_cls, preds in prediction["prediction"].items():
                    for pred in preds:
                        split_preds.append([pred_cls] + pred)

                split_preds.sort(key=lambda x: x[1])

                is_test_file = prediction["is_test_file"]

                for record_idx, split_pred in enumerate(split_preds):

                    logging.info(f"Saving prediction for {dataset_id}, {ibdoc_filename}, {record_idx}")

                    prediction_writer.add_prediction(
                        dataset,
                        ibdoc_filename,
                        record_idx,
                        PredictionResultDict(
                            annotated_class_name=split_pred[0],
                            page_start=split_pred[1] - 1,
                            page_end=split_pred[2] - 1,
                            classification_confidence=split_pred[3],
                            split_confidence=split_pred[4],
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
        self.set_status({"predictions_uuid": uuid.uuid4().hex})

    def generate_classifier(self, label_names):
        logging.info("Generating the classifier module for this model...")

        try:
            ib_model_path = os.path.join(self.ib_save_path, "artifact")
            write_classifier_module(self.artifacts_context, ib_model_path, label_names, self.model_name)
            logging.info("Finished generating the Classifier module")
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Skipped classifier module generation due to an error: {e}")

    def write_epoch_summary(self):
        logging.info("Metrics over Epochs")
        for epoch, met in enumerate(self.evaluation_results):
            logging.info(f"Epoch {epoch} ^")
            split_metrics = {k: v for k, v in met["splitter_metrics"].items() if k != "accuracy"}
            logging.info(pd.DataFrame(split_metrics))
            class_metrics = {k: v for k, v in met["classifier_metrics"].items() if k != "accuracy"}
            logging.info(pd.DataFrame(class_metrics))

    def on_predict(self, args, state, control, **kwargs):
        # called after the training finish
        predictions = kwargs["metrics"]["predict_predictions"]
        # FINALIZE STEPS
        self.write_metrics()
        self.write_predictions(predictions)

        id2label = kwargs["model"].config.id2label
        label_names = [id2label[idx] for idx in range(0, len(id2label))]
        self.generate_classifier(label_names)
        self.move_data_to_ib()
        self.write_epoch_summary()
