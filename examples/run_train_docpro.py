import os
from typing import Optional, Dict, Any

from ibformers.trainer.ib_utils import InstabaseSDKDummy
from ibformers.trainer.train import run_train_doc_pro
from instabase.model_training_tasks.jobs import JobMetadataClient


# below is for debugging with running locally
# this code is not reached via model service as it is directly calling run_train fn
class DummyJobStatus:
    def __init__(self):
        pass

    def update_message(self, message: Optional[str]) -> None:
        pass

    def update_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        pass


if __name__ == "__main__":

    '/Users/rafalpowalski/python/annotation/UberEatsDataset'

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

    dataset_filename = '/Users/rafalpowalski/python/annotation/UberEatsDataset'
    save_path = '/Users/rafalpowalski/python/models/test_model'
    sdk = InstabaseSDKDummy(None, "user")
    run_train_doc_pro(
        hyperparams=hyperparams,
        dataset_paths=[dataset_filename],
        save_path=save_path,
        extraction_class_name='UberEats',
        file_client=sdk,
        username='user',
        job_metadata_client=DummyJobStatus(),
        mount_details=None,
        model_name='CustomModel',
    )
