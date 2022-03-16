import logging
import os
import shutil
import traceback
import uuid
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from ibformers.trainer.hp_search.optimize import get_study_summary
from ibformers.trainer.ib_utils import InstabaseSDK, _abspath, upload_dir
from ibformers.trainer.refiner_module_generator import write_refiner_program
from ibformers.utils.print_dir import print_dir
from ibformers.utils.version_info import SHOULD_FORMAT
from ibformers.utils.zip_dir import zip_dir

from instabase.dataset_utils.sdk import DatasetSDK
from instabase.dataset_utils.shared_types import (
    IndexedWordDict,
    PredictionFieldDict,
    PredictionInstanceDict,
    PredictionResultDict,
)
from instabase.training_utils.model_artifact import (
    ModelArtifactContext,
    ModelArtifactMetricsWriter,
    ModelArtifactPredictionsWriter,
    TableMetric,
    ValueMetric,
)
from transformers import TrainerCallback, TrainerState


class DocProCallback(TrainerCallback):
    """
    Handles events specific to doc pro
    """

    # list of files/dirs which will not be copied into the package
    ibformers_do_not_copy = ["hf_token.py"]

    # For UX purposes, avoid setting progress bar to 100% until upload finishes
    PROGRESS_BAR_LIMIT: float = 0.95
    LOGGING_STEPS: int = 10
    MAX_HP_TRIALS_TO_DISPLAY = 10

    def __init__(
        self,
        dataset_list: List[DatasetSDK],
        artifacts_context: ModelArtifactContext,
        extraction_class_name: Optional[str],
        job_metadata_client: "JobMetadataClient",  # type: ignore
        ibsdk: InstabaseSDK,
        username: str,
        mount_details: Dict,
        model_name: str,
        ib_save_path: str,
        log_metrics_to_metadata: bool = False,
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
        :param log_metrics_to_metadata: wheter to log metrics to metadata of job_metadata_client

        """
        self.log_metrics_to_metadata = log_metrics_to_metadata
        self.extraction_class_name = extraction_class_name
        self.dataset_list = dataset_list
        self.artifacts_context = artifacts_context
        self.ib_save_path = ib_save_path
        self.model_name = model_name
        self.mount_details = mount_details
        self.username = username
        self.job_metadata_client = job_metadata_client
        self.hyperparamsearch_results = None
        self.evaluation_results = []
        self.prediction_results = None
        self.ibsdk = ibsdk
        self.job_status = {}
        self.hp_search_progress_end = 0.0
        self.hp_search_progress_step = 0.0
        self.metrics_writer = ModelArtifactMetricsWriter(artifacts_context)
        self.prediction_writer = ModelArtifactPredictionsWriter(artifacts_context)
        self.save_folder = artifacts_context.tmp_dir.name
        self.save_model_dir = os.path.join(artifacts_context.artifact_path, f"src/py/{model_name}/model_data")
        self.id_to_dataset = {dataset.metadata["id"]: dataset for dataset in dataset_list}
        # paths to zipped files, must be relative to self.save_folder
        self._zipped: List[str] = []

    def copy_library_src_to_package(self):
        # copy ibformers lib into the package

        ibformers_path = Path(_abspath("")).parent
        assert ibformers_path.name == "ibformers", f"ibformers_path is wrong. Path: {ibformers_path}"
        # TODO(rafal): once ibformers is converted to relative imports copy ibformers into py/package_name/ibformers dir
        # py directory location
        py_directory = Path(self.artifacts_context.artifact_path) / "src" / "py"
        zip_dir(
            ibformers_path,
            py_directory / "ibformers.zip",
            ignore_files=self.ibformers_do_not_copy,
            ignore_dirs=["__pycache__"],
            ignore_hidden=True,
        )
        self._zipped.append(str((py_directory / "ibformers.zip").relative_to(self.save_folder)))

    def move_data_to_ib(self):
        self.copy_library_src_to_package()

        logging.info("Final state of the Model Artifact folder structure:")
        print_dir(self.save_folder)

        self.update_message("Uploading model.", log=True)
        upload_dir(
            sdk=self.ibsdk,
            local_folder=self.save_folder,
            remote_folder=self.ib_save_path,
            mount_details=self.mount_details,
        )
        logging.info("Uploaded model")
        if self._zipped:
            logging.info("Unzipping model files")
            for path in self._zipped:
                logging.debug(f"Unzipping file {os.path.join(self.ib_save_path, path)}")
                full_ib_path = os.path.join(self.ib_save_path, path)
                self.ibsdk.unzip(full_ib_path, os.path.splitext(full_ib_path)[0], remove=True)
            logging.info("Unzipped model files")

        self.set_status({"progress": 1.0})

    def update_message(self, message: str, log: bool = False):
        if log:
            logging.info(message)
        self.job_metadata_client.update_message(message)

    def set_status(self, new_status: Dict):
        self.job_status.update(new_status)
        self.job_metadata_client.update_metadata(self.job_status)

    def set_training_progress(self, state: TrainerState):
        if state.global_step % self.LOGGING_STEPS == 0:
            progress = 0.0
            if state.max_steps != 0:
                progress_range = self.PROGRESS_BAR_LIMIT - self.hp_search_progress_end
                progress_increase = progress_range * (state.global_step / state.max_steps)
                progress = self.hp_search_progress_end + progress_increase
            self.set_status({"progress": progress})

    def set_hp_search_progress(self, state: TrainerState):
        if state.global_step % self.LOGGING_STEPS == 0:
            trial_number = int(state.trial_name.split("-")[1])
            trial_base_progress = trial_number * self.hp_search_progress_step
            progress_increase = 0.0
            if state.max_steps != 0:
                progress_increase = self.hp_search_progress_step * (state.global_step / state.max_steps)
            progress = trial_base_progress + progress_increase
            self.set_status({"progress": progress})

    def set_progress(self, state: TrainerState):
        if state.is_hyper_param_search:
            self.set_hp_search_progress(state)
        else:
            self.set_training_progress(state)

    def on_hyperparam_search_start(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.job_metadata_client.update_message("Parameter tuning in progress.")
            total_train_jobs = args.hp_search_num_trials + 1
            self.hp_search_progress_end = self.PROGRESS_BAR_LIMIT * args.hp_search_num_trials / total_train_jobs
            self.hp_search_progress_step = self.hp_search_progress_end / args.hp_search_num_trials

    def on_hyperparam_search_end(self, args, state, control, study, **kwargs):
        if state.is_local_process_zero:
            study_summary = get_study_summary(study, args)
            all_trials = study_summary["trials_summary"]
            self.hyperparamsearch_results = sorted(
                all_trials, key=itemgetter("objective"), reverse=not args.hp_search_do_minimize_objective
            )

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.set_progress(state)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if not state.is_hyper_param_search:
                self.job_metadata_client.update_message("Model training in progress.")
            self.set_progress(state)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.set_progress(state)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero and not state.is_hyper_param_search:
            # workaround for missing on_predict callback in the transformers TrainerCallback
            if "predict_loss" in kwargs["metrics"]:
                self.on_predict(args, state, control, **kwargs)
            elif "eval_loss" in kwargs["metrics"]:
                metrics = {}
                metrics["exact_match"] = kwargs["metrics"]["eval_exact_match"]
                metrics["precision"] = kwargs["metrics"]["eval_precision"]
                metrics["recall"] = kwargs["metrics"]["eval_recall"]
                metrics["f1"] = kwargs["metrics"]["eval_f1"]
                if self.log_metrics_to_metadata:
                    self.set_status({"evaluation_results": metrics})

                self.evaluation_results.append(metrics)

            elif "test_eval_loss" in kwargs["metrics"]:
                # write final evaluation on test-marked documents.
                metrics = {}
                metrics["exact_match"] = kwargs["metrics"]["test_eval_exact_match"]
                metrics["precision"] = kwargs["metrics"]["test_eval_precision"]
                metrics["recall"] = kwargs["metrics"]["test_eval_recall"]
                metrics["f1"] = kwargs["metrics"]["test_eval_f1"]
                if self.log_metrics_to_metadata:
                    self.set_status({"evaluation_results": metrics})
                self.evaluation_results.append(metrics)

            else:
                pass

    def write_metrics(self):
        logging.info("Writing metrics for this training run...")
        metrics_writer = self.metrics_writer
        overall_accuracy = "Unknown"
        try:
            # get the last evalutation metrics
            evaluation_metrics = self.evaluation_results[-1]
            if evaluation_metrics:

                # First add the table
                headers, rows = self.get_field_level_table_data(evaluation_metrics)

                metrics_writer.add_table_metric(
                    TableMetric(
                        title="Field-level Metrics",
                        subtitle="Accuracy scores for individual fields learned by the model",
                        headers=["Field Name"] + headers,
                        rows=rows,
                    )
                )

                # Add hp search results
                if (
                    self.hyperparamsearch_results is not None
                    and len(self.hyperparamsearch_results) > 0
                    and SHOULD_FORMAT  # we cannot display this table without self-formatting
                ):
                    hp_headers, hp_rows = self.get_hp_search_table_data()

                    metrics_writer.add_table_metric(
                        TableMetric(
                            title="Hyperparameter search results",
                            subtitle=f"Top {len(hp_rows)} runs from hyperparameter search, sorted by the objective "
                            f"value on validation dataset",
                            headers=hp_headers,
                            rows=hp_rows,
                        )
                    )

                raw_recalls = evaluation_metrics["recall"].values()
                recalls = [x for x in raw_recalls if x != "NAN"]
                f1_scores = [
                    f1 if f1 != "NAN" else 0.0
                    for (f1, recall) in zip(evaluation_metrics["f1"].values(), raw_recalls)
                    if recall != "NAN"
                ]
                precisions = [
                    pr if pr != "NAN" else 0.0
                    for (pr, recall) in zip(evaluation_metrics["precision"].values(), raw_recalls)
                    if recall != "NAN"
                ]

                # Then add high-level metrics
                avg_f1 = "{:.2f}%".format(sum(f1_scores) * 100.0 / float(len(f1_scores))) if f1_scores else "N/A"
                overall_accuracy = (
                    "{:.2f}%".format(sum(f1_scores) * 100.0 / float(len(f1_scores))) if f1_scores else "Unknown"
                )
                metrics_writer.add_high_level_metric(
                    ValueMetric(
                        title="Macro F1",
                        subtitle="Average F1 score across all fields",
                        value=avg_f1,
                        tag_text="ACCURACY",
                        tag_type="INFO",
                    )
                )
                avg_precision = (
                    "{:.2f}%".format(sum(precisions) * 100.0 / float(len(precisions))) if precisions else "N/A"
                )
                metrics_writer.add_high_level_metric(
                    ValueMetric(
                        title="Macro Precision",
                        subtitle="The average precision across all fields",
                        value=avg_precision,
                        tag_text="ACCURACY",
                        tag_type="INFO",
                    )
                )
                avg_recall = "{:.2f}%".format(sum(recalls) * 100.0 / float(len(recalls))) if recalls else "N/A"
                metrics_writer.add_high_level_metric(
                    ValueMetric(
                        title="Average Recall",
                        subtitle="The average recall across all fields",
                        value=avg_recall,
                        tag_text="ACCURACY",
                        tag_type="INFO",
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
        self.set_status({"accuracy": overall_accuracy})

    def write_predictions(self, predictions_dict, zip_predictions=False):
        # Now write the predictions
        prediction_writer = self.prediction_writer

        self.update_message("Writing prediction results.")
        logging.info("Writing prediction results to IB")
        try:
            for record_name, record_value in predictions_dict.items():
                # TODO(VONTELL) - rn we are using the field name directly.
                # Using the vars below, we could get the actual ids...
                dataset_id = record_name[:36]
                dataset = self.id_to_dataset[dataset_id]
                # This is because the first 37 chars [0]-[36] are reserved for the UUID (36), and a dash (-)
                ibdoc_filename = record_name[37 : record_name.index(".ibdoc") + len(".ibdoc")]
                # This grabs the record index at the end of the file name
                record_idx = int(record_name[len(dataset_id) + len(ibdoc_filename) + 2 : -1 * len(".json")])
                logging.info(f"Saving prediction for {dataset_id}, {ibdoc_filename}, {record_idx}")

                # Currently, our models only outputs a single entity
                # Predictions, however, can support multiple
                fields = []
                is_test_file = record_value["is_test_file"]
                record_entities = record_value["entities"]
                for field_name, field_value in record_entities.items():
                    indexed_words = [
                        IndexedWordDict(line_index=w["word_line_idx"], word_index=w["word_in_line_idx"])
                        for w in field_value["words"]
                    ]
                    predictions = (
                        [
                            PredictionInstanceDict(
                                avg_confidence=field_value["avg_confidence"],
                                value=field_value["text"],
                                words=indexed_words,
                            )
                        ]
                        if len(indexed_words) > 0
                        else []
                    )

                    fields.append(PredictionFieldDict(field_name=field_name, annotations=predictions))
                prediction_writer.add_prediction(
                    dataset,
                    ibdoc_filename,
                    record_idx,
                    PredictionResultDict(annotated_class_name=self.extraction_class_name, fields=fields),
                    is_test_file,
                )
        except Exception as e:
            logging.error(traceback.format_exc())
        try:
            logging.info("Writing predictions for this training run...")
            prediction_writer.write()
            # zip folder with predictions and remove it
            if zip_predictions:
                _path = Path(prediction_writer.context.predictions_path)
                zip_dir(_path, _path.with_suffix(".zip"))
                shutil.rmtree(_path)
                self._zipped.append(str(_path.with_suffix(".zip").relative_to(self.save_folder)))
        except Exception as e:
            logging.error("Could not write prediction")
            logging.error(traceback.format_exc())
        self.set_status({"predictions_uuid": uuid.uuid4().hex})

    def generate_refiner(self, label_names):
        self.update_message("Generating Refiner module.", log=True)

        try:
            ib_model_path = os.path.join(self.ib_save_path, "artifact")
            dev_path = os.path.join(self.dataset_list[0].dataset_path, self.dataset_list[0].metadata["docs_path"])
            write_refiner_program(self.artifacts_context, ib_model_path, label_names, self.model_name, dev_path)
            logging.info("Finished generating the Refiner module")
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Skipped Refiner module generation due to an error: {e}")

    def write_epoch_summary(self):
        logging.info("Metrics over Epochs")
        for epoch, metrics in enumerate(self.evaluation_results):
            logging.info(f"Epoch {epoch} ^")
            logging.info(pd.DataFrame(metrics))

    def on_predict(self, args, state, control, **kwargs):
        # called after the training finish
        predictions = kwargs["metrics"]["predict_predictions"]
        # FINALIZE STEPS
        self.write_metrics()
        self.write_predictions(predictions, zip_predictions=True)

        id2label = kwargs["model"].config.ib_id2label
        if id2label[0] != "O":
            logging.error(f"0 index for label should be asigned to O class. Got: {id2label[0]}")
        label_names = [id2label[idx] for idx in range(1, len(id2label))]
        self.generate_refiner(label_names)
        self.move_data_to_ib()
        self.write_epoch_summary()

    def get_field_level_table_data(self, evaluation_metrics):
        headers = list(evaluation_metrics.keys())
        all_rows = set()
        for column in evaluation_metrics:
            all_rows.update(list(evaluation_metrics[column].keys()))
        sorted_rows = list(all_rows)
        sorted_rows.sort()
        rows = []
        for row_key in sorted_rows:
            values = [evaluation_metrics[c].get(row_key, "") for c in headers]
            if SHOULD_FORMAT:
                values = [format(v, ".2%") for v in values]
            rows.append([row_key, *values])
        return headers, rows

    def get_hp_search_table_data(self):

        param_names = list(self.hyperparamsearch_results[0]["params"].keys())
        headers = ["Trial number"] + param_names + ["Objective value"]
        values = [
            [str(trial["number"])]
            + [str(trial["params"][param_name]) for param_name in param_names]
            + [format(trial["objective"], ".4f")]
            for trial in self.hyperparamsearch_results
        ]
        return headers, values
