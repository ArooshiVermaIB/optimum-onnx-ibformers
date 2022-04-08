import logging
import traceback
from typing import Dict, List, Optional

from ibformers.callbacks.extraction import DocProCallback
from ibformers.trainer.ib_utils import InstabaseSDK
from instabase.dataset_utils.sdk import DatasetSDK
from instabase.training_utils.model_artifact import (
    ModelArtifactContext,
    ValueMetric,
)

HIGH_LEVEL_METRICS = [
    {
        "raw_name": "grits_loc",
        "title": "Cell Location Similarity",
        "subtitle": "F1-like similarity to ground truth based on cell location",
        "tag_text": "ACCURACY",
        "tag_type": "INFO",
    },
    {
        "raw_name": "grits_loc_rowbased",
        "title": "Row Location Similarity",
        "subtitle": "F1-like similarity to ground truth based on row location",
        "tag_text": "ACCURACY",
        "tag_type": "INFO",
    },
    {
        "raw_name": "grits_loc_colbased",
        "title": "Column Location Similarity",
        "subtitle": "F1-like similarity to ground truth based on column location",
        "tag_text": "ACCURACY",
        "tag_type": "INFO",
    },
    {
        "raw_name": "grits_cont",
        "title": "Cell Content Similarity",
        "subtitle": "F1-like similarity to ground truth based on cell content",
        "tag_text": "ACCURACY",
        "tag_type": "INFO",
    },
    {
        "raw_name": "grits_cont_rowbased",
        "title": "Row Content Similarity",
        "subtitle": "F1-like similarity to ground truth based on row content",
        "tag_text": "ACCURACY",
        "tag_type": "INFO",
    },
    {
        "raw_name": "grits_cont_colbased",
        "title": "Column Content Similarity",
        "subtitle": "F1-like similarity to ground truth based on column content",
        "tag_text": "ACCURACY",
        "tag_type": "INFO",
    },
]


class IbTableExtractionCallback(DocProCallback):
    """
    Handles IB-related events for table extraction
    """

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
            log_metrics_to_metadata=log_metrics_to_metadata,
        )

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            # workaround for missing on_predict callback in the transformers TrainerCallback
            if "predict_loss" in kwargs["metrics"]:
                self.on_predict(args, state, control, **kwargs)
            elif "eval_loss" in kwargs["metrics"]:
                metrics = kwargs["metrics"]["eval_metrics"]
                self.set_status({"evaluation_results": metrics, "progress": state.global_step / state.max_steps})

                self.evaluation_results.append(metrics)
            elif "test_eval_loss" in kwargs["metrics"]:
                metrics = kwargs["metrics"]["test_eval_metrics"]
                self.set_status({"evaluation_results": metrics})
                self.evaluation_results.append(metrics)
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

                # Then add high-level metrics
                for metric in HIGH_LEVEL_METRICS:
                    metric_value = evaluation_metrics[metric]
                    metrics_writer.add_high_level_metric(
                        ValueMetric(
                            title=metric["title"],
                            subtitle=metric["subtitle"],
                            value=f"{metric_value:.2%}",
                            tag_text=metric["tag_text"],
                            tag_type=metric["tag_type"],
                        )
                    )
            overall_accuracy = evaluation_metrics["grits_cont"]
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

    def on_predict(self, args, state, control, **kwargs):
        # called after the training finish
        predictions = kwargs["metrics"]["predict_predictions"]
        # FINALIZE STEPS
        self.write_metrics()
        self.write_predictions(predictions)

        # TODO: generate refiner
        # id2label = kwargs["model"].config.id2label
        # label_names = [id2label[idx] for idx in range(1, len(id2label))]
        # self.generate_refiner(label_names)
        self.move_data_to_ib()
        self.write_epoch_summary()
