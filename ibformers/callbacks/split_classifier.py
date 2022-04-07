import logging
import os
import traceback
import uuid
from typing import Dict, List

import pandas as pd
from ibformers.callbacks.classifier import DocProClassificationCallback
from ibformers.trainer.ib_utils import InstabaseSDK
from ibformers.trainer.splitclassifier_module_generator import write_classifier_module
from instabase.dataset_utils.sdk import DatasetSDK
from instabase.dataset_utils.shared_types import PredictionResultDict
from instabase.training_utils.model_artifact import (
    ModelArtifactContext,
    TableMetric,
    ValueMetric,
)


class SplitClassifierCallback(DocProClassificationCallback):
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

        super().__init__(
            dataset_list=dataset_list,
            artifacts_context=artifacts_context,
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
            elif "test_eval_loss" in kwargs["metrics"]:
                metrics = {}
                splitter_metrics = kwargs["metrics"]["test_eval_splitter_metrics"]
                classifier_metrics = kwargs["metrics"]["test_eval_classifier_metrics"]
                metrics["splitter_metrics"] = splitter_metrics
                metrics["classifier_metrics"] = classifier_metrics

                # Had to duplicate this so that CI tests work, since it currently supports only
                # two levels of nesting
                metrics.update({f"splitter_{k}": v for k, v in splitter_metrics.items()})
                metrics.update({f"classifier_{k}": v for k, v in classifier_metrics.items()})
                self.set_status({"evaluation_results": metrics})

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

                # First add table metrics

                # Splitter table
                splitter_metrics = evaluation_metrics["splitter_metrics"]
                splitter_headers, splitter_rows = self.get_classifier_results_table_data(splitter_metrics)
                metrics_writer.add_table_metric(
                    TableMetric(
                        title="Splitter Metrics",
                        subtitle="Accuracy scores for split/no-split learned by the model",
                        headers=splitter_headers,
                        rows=splitter_rows,
                    )
                )

                # Classifier table
                classifier_metrics = evaluation_metrics["classifier_metrics"]
                classifier_headers, classifier_rows = self.get_classifier_results_table_data(classifier_metrics)
                metrics_writer.add_table_metric(
                    TableMetric(
                        title="Classifier Metrics",
                        subtitle="Accuracy scores for classes learned by the model",
                        headers=classifier_headers,
                        rows=classifier_rows,
                    )
                )

                # Now add high level metrics

                # Add classifier accuracy
                classifier_accuracy = classifier_metrics["accuracy"]
                metrics_writer.add_high_level_metric(
                    ValueMetric(
                        title="Accuracy for classifier",
                        subtitle="Accuracy",
                        value=f"{classifier_accuracy:.2%}",
                        tag_text="ACCURACY",
                        tag_type="INFO",
                    )
                )

                # Add splitter accuracy
                splitter_accuracy = splitter_metrics["accuracy"]
                metrics_writer.add_high_level_metric(
                    ValueMetric(
                        title="Accuracy for splitter",
                        subtitle="Accuracy",
                        value=f"{splitter_accuracy:.2%}",
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
        overall_accuracy = f"{(splitter_accuracy + classifier_accuracy) / 2.0 :.2%}"
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
        label_names = list(id2label.values())
        self.generate_classifier(label_names)
        self.move_data_to_ib()
        self.write_epoch_summary()
