import os
from pathlib import Path
from typing import Optional, Dict, Any

from ibformers.trainer.ib_utils import run_train_annotator
from instabase.model_training_tasks.jobs import JobMetadataClient


# below is for debugging with running locally
# this code is not reached via model service as it is directly calling run_train fn
class DummyJobStatus:
    def __init__(self):
        self.job_id = 1

    def update_message(self, message: Optional[str]) -> None:
        pass

    def update_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        pass


class InstabaseSDKDummy:
    def __init__(self, file_client: Any, username: str):
        # these will be ignored
        self.file_client = file_client
        self.username = username

    def ibopen(self, path: str, mode: str = "r") -> Any:
        return open(path, mode)

    def read_file(self, file_path: str) -> str:
        with open(file_path, 'r') as f:
            return f.read()

    def write_file(self, file_path: str, content: str):
        # mkdir
        Path(file_path).parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "wb") as f:
            f.write(content)


if __name__ == "__main__":

    hyperparams = {
        "adam_epsilon": 1e-8,
        "batch_size": 8,
        "chunk_size": 512,
        "epochs": 3,
        "learning_rate": 5e-05,
        "loss_agg_steps": 2,
        "max_grad_norm": 1.0,
        "optimizer_type": "AdamW",
        "scheduler_type": "constant_schedule_with_warmup",
        "stride": 64,
        "use_gpu": True,
        "use_mixed_precision": False,
        "warmup": 0.0,
        "weight_decay": 0,
        "model_name": "microsoft/layoutlm-base-uncased",
    }

    dataset_filename = '/Users/rafalpowalski/python/annotation/receipts/Receipts.ibannotator'
    save_path = '/Users/rafalpowalski/python/models/test_model'
    sdk = InstabaseSDKDummy(None, "user")
    run_train_annotator(hyperparams, dataset_filename, save_path, sdk, 'user', DummyJobStatus())
