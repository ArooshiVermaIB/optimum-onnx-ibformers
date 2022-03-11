from transformers import EarlyStoppingCallback
from transformers.utils import logging

logger = logging.get_logger(__name__)


class IbEarlyStoppingCallback(EarlyStoppingCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            # ignore the warning for final eval and test eval
            if not (f"final_{metric_to_check}" in metrics or f"test_{metric_to_check}" in metrics):
                logger.warning(
                    f"early stopping required metric_for_best_model, but did not find {metric_to_check} "
                    f"so early stopping is disabled"
                )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
            best_epoch_number = state.epoch - self.early_stopping_patience
            logger.info(
                f"Training stopped due to the early stopping after {state.epoch} epochs. "
                f"Optimal number of epochs was determined to be {best_epoch_number}"
            )
