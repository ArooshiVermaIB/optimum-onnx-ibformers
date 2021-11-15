from transformers import ProgressCallback

from ibformers.callbacks.utils import rewrite_logs


class CustomProgressCallback(ProgressCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = rewrite_logs(logs)
        super().on_log(args, state, control, logs, **kwargs)
