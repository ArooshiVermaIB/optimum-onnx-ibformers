import collections
import sys
from logging import StreamHandler
from typing import Optional, List, Dict
import numpy as np
import torch
from datasets import IterableDataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers.integrations import WandbCallback, is_wandb_available
from transformers.trainer import Trainer
from transformers import EvalPrediction, is_torch_tpu_available, AdamW
from transformers.file_utils import (
    is_in_notebook,
    is_apex_available,
    is_datasets_available,
    is_training_run_on_sagemaker,
)
from transformers.trainer_pt_utils import (
    nested_truncate,
    IterableDatasetShard,
    nested_concat,
    nested_numpify,
    find_batch_size,
)
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize, PredictionOutput
from transformers.utils import logging

from ibformers.callbacks.wandb import ExtendedWandbCallback

logger = logging.get_logger(__name__)

_is_torch_generator_available = True
_is_native_amp_available = True


if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl


if is_training_run_on_sagemaker():
    logging.add_handler(StreamHandler(sys.stdout))


class IbTrainer(Trainer):
    """
    Copied from transformers to include few modifications inside the training loop
    Old trainer will be changed once transformers will be updated on ib
    """

    def __init__(self, *args, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Add replace the default wandb callback with the custom one that logs more data'
        self.post_process_function = post_process_function
        self.test_dataset = None
        self._update_callbacks()

    def _update_callbacks(self) -> None:
        if is_wandb_available():
            old_callback = self.pop_callback(WandbCallback)
            if old_callback is not None:
                self.add_callback(ExtendedWandbCallback)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Add support for class weights
        """
        if self.args.class_weights > 1 and "labels" in inputs:
            labels = inputs.pop("labels")
            outputs = model(**inputs)

            # class_weights parameter is used to enlarge weights of tokens corresponding to entity values
            # that might help model to converge faster
            class_num = self.model.config.num_labels
            class_weights = [1.0] + [self.args.class_weights] * (class_num - 1)
            loss_fct = CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(labels.device))
            attention_mask = inputs.get("attention_mask", None)
            logits = outputs["logits"]
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, class_num)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, class_num), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:

        if self.args.max_no_annotation_examples_share < 1.0:
            # Build the sampler.
            max_no_annotation_examples_share = self.args.max_no_annotation_examples_share
            col_names = self.train_dataset.column_names
            # QA task
            if "start_positions" in col_names and "end_positions" in col_names:
                labels = self.train_dataset["start_positions"]
            # Token classification task
            elif "labels" in col_names:
                labels = self.train_dataset["labels"]
                if not isinstance(labels[0], list):
                    raise ValueError("max_no_annotation_examples_share expect token_classification task")
            else:
                raise ValueError(
                    "max_no_annotation_examples_share parameter is supported only with "
                    "token_classification and qa pipelines. It require specific labels column"
                )

            has_annotations = np.array([(np.array(lab) > 0).sum().item() > 0 for lab in labels])
            ratio_of_unannotated_to_annotated = max_no_annotation_examples_share / (
                1 - max_no_annotation_examples_share
            )
            num_annotations = int(has_annotations.sum())
            num_no_annotations = len(has_annotations) - num_annotations
            max_no_annotation = round(ratio_of_unannotated_to_annotated * has_annotations.sum())
            num_samples = num_annotations + max_no_annotation

            if max_no_annotation < num_no_annotations:
                logger.warning(
                    f"Limitting number of training chunks from {len(has_annotations)} "
                    f"to {num_samples} due to high ratio of no annotation chunks"
                )
                if self.args.group_by_length:
                    logger.error(
                        "group_by_length param is not supported together with max_no_annotation_examples_share. "
                        "Future processing will ignore group_by_length"
                    )
                if self.args.world_size > 1:
                    # TODO: implement DistributedWeightedRandomSampler
                    raise ValueError("max_no_annotation_examples_share is not supported in the multi-gpu setting")

                generator = None
                if self.args.world_size <= 1 and _is_torch_generator_available:
                    generator = torch.Generator()
                    generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

                no_ann_weight: float = max_no_annotation / num_no_annotations
                weights = [1.0 if is_ann else no_ann_weight for is_ann in has_annotations]

                return WeightedRandomSampler(
                    replacement=False, weights=weights, num_samples=num_samples, generator=generator
                )
            else:
                return super()._get_train_sampler()
        else:
            return super()._get_train_sampler()

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            # MODIFICATION - pass eval_dataset to metric computing in order to get document level predictions
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels),
                self.eval_dataset if metric_key_prefix in {"eval", "final_eval"} else self.test_dataset,
            )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
    ) -> PredictionOutput:
        """
        run predict method from original trainer but call on_predict callback in the end
        """
        # keep the original reference to test_dataset as test_dataloader removes columns not used during model trainn
        self.test_dataset = test_dataset
        output = super().predict(test_dataset, ignore_keys, metric_key_prefix)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Method removes predictions from logging
        """
        logs_mod = {k: v for k, v in logs.items() if not k.endswith("predictions")}
        super().log(logs_mod)
