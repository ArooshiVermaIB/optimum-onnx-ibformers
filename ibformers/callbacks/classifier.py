import logging
import os
import traceback
import uuid
from typing import Dict, List

import pandas as pd
from ibformers.callbacks.extraction import DocProCallback
from ibformers.trainer.classifier_module_generator import write_classifier_module
from ibformers.trainer.ib_utils import InstabaseSDK
from ibformers.utils.version_info import SHOULD_FORMAT
from instabase.dataset_utils.sdk import DatasetSDK
from instabase.dataset_utils.shared_types import PredictionResultDict
from instabase.training_utils.model_artifact import (
    ModelArtifactContext,
    TableMetric,
    ValueMetric,
)


class DocProClassificationCallback(DocProCallback):
    """
    Handles events specific to doc pro
    """

    # list of files/dirs which will not be copied into the package
    ibformers_do_not_copy = ["hf_token.py"]

    def __init__(
        self,
        dataset_list: List[DatasetSDK],
        artifacts_context: ModelArtifactContext,
        # extraction_class_name: Optional[str],
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
        :param extraction_class_name: name of the extracted class
        :param job_metadata_client: client used by callback to log progress/status of the training
        :param ibsdk: sdk used to transfer files to IB fs
        :param username: username of user who run the training job
        :param mount_details: optional details of s3 mount
        :param model_name: name of the model used in front end
        :param ib_save_path: ib location of the training job output

        """
        super().__init__(
            dataset_list=dataset_list,
            artifacts_context=artifacts_context,
            extraction_class_name=None,
            job_metadata_client=job_metadata_client,
            ibsdk=ibsdk,
            username=username,
            mount_details=mount_details,
            model_name=model_name,
            ib_save_path=ib_save_path,
        )

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            # workaround for missing on_predict callback in the transformers TrainerCallback
            if "predict_loss" in kwargs["metrics"]:
                self.on_predict(args, state, control, **kwargs)
            elif "eval_loss" in kwargs["metrics"]:
                classifier_metrics = kwargs["metrics"]["eval_metrics"]
                self.set_status(
                    {"evaluation_results": classifier_metrics, "progress": state.global_step / state.max_steps}
                )

                self.evaluation_results.append(classifier_metrics)
            elif "test_eval_loss" in kwargs["metrics"]:
                classifier_metrics = kwargs["metrics"]["test_eval_metrics"]
                self.set_status({"evaluation_results": classifier_metrics})
                self.evaluation_results.append(classifier_metrics)
            else:
                # ignore last evaluation call
                pass

    def write_metrics(self):
        logging.info("Writing metrics for this training run...")
        metrics_writer = self.metrics_writer
        overall_accuracy = "Unknown"
        try:
            evaluation_metrics = self.job_status.get("evaluation_results")
            if evaluation_metrics:

                # First add the table
                classifier_headers, classifier_rows = self.get_classifier_results_table_data(evaluation_metrics)

                metrics_writer.add_table_metric(
                    TableMetric(
                        title="Splitter Metrics",
                        subtitle="Accuracy scores for split/no-split learned by the model",
                        headers=classifier_headers,
                        rows=classifier_rows,
                    )
                )

                accuracy = evaluation_metrics["accuracy"]
                overall_accuracy = "{:.2f}".format(accuracy * 100)

                precision = evaluation_metrics["precision"]
                recall = evaluation_metrics["recall"]
                f1 = evaluation_metrics["f1"]

                # Then add high-level metrics
                accuracy = evaluation_metrics["accuracy"]
                overall_accuracy = "{:.2f}".format(accuracy * 100)

                metrics_writer.add_high_level_metric(
                    ValueMetric(
                        title="Macro F1",
                        subtitle="Average F1 score across all classes",
                        value=f"{f1.get('macro avg', 0.0):.2%}",
                        tag_text="ACCURACY",
                        tag_type="INFO",
                    )
                )

                metrics_writer.add_high_level_metric(
                    ValueMetric(
                        title="Macro Precision",
                        subtitle="The average precision across all classes",
                        value=f"{precision.get('macro avg', 0.0):.2%}",
                        tag_text="ACCURACY",
                        tag_type="INFO",
                    )
                )
                metrics_writer.add_high_level_metric(
                    ValueMetric(
                        title="Average Recall",
                        subtitle="The average recall across all classes",
                        value=f"{recall.get('macro avg', 0.0):.2%}",
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

    def get_classifier_results_table_data(self, evaluation_metrics):
        classifier_table = pd.DataFrame({k: v for k, v in evaluation_metrics.items() if k != "accuracy"})
        classifier_table = classifier_table[["f1", "precision", "recall", "support"]]
        if SHOULD_FORMAT:
            classifier_table[["f1", "precision", "recall"]] = classifier_table[["f1", "precision", "recall"]].applymap(
                lambda x: self.maybe_format_as_percent(x)
            )
        classifier_records = classifier_table.to_records(index=True)

        classifier_headers = list(classifier_records.dtype.names)
        classifier_headers[0] = "Class Type"
        classifier_rows = classifier_records.tolist()
        return classifier_headers, classifier_rows

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
                ibdoc_filename = record_name[37 : record_name.index(".ibdoc") + len(".ibdoc")]
                # This grabs the record index at the end of the file name
                record_idx = int(record_name[len(dataset_id) + len(ibdoc_filename) + 2 : -1 * len(".json")])
                logging.info(f"Saving prediction for {dataset_id}, {ibdoc_filename}, {record_idx}")

                # Currently, our models only outputs a single entity
                # Predictions, however, can support multiple
                is_test_file = record_value["is_test_file"]

                prediction_writer.add_prediction(
                    dataset,
                    ibdoc_filename,
                    record_idx,
                    PredictionResultDict(
                        annotated_class_name=record_value["class_label"],
                        classification_confidence=record_value["class_confidence"],
                        page_start=record_value["page_start"],
                        page_end=record_value["page_end"],
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

    def on_predict(self, args, state, control, **kwargs):
        # called after the training finish
        # FINALIZE STEPS
        self.write_metrics()
        self.write_predictions(kwargs["metrics"]["predict_predictions"])
        id2label = kwargs["model"].config.id2label
        label_names = list(id2label.values())
        self.generate_classifier(label_names)
        self.move_data_to_ib()
        self.write_epoch_summary()
