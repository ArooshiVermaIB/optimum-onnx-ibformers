from pathlib import Path
import os
from typing import Optional, Dict, Any
from ibformers.trainer.docpro_utils import run_train_both_classification
import fire
import zipfile
from tqdm import tqdm

# SCRIPT USED FOR DEBUGGING WITH LOCAL RUNS
class DummyJobStatus:
    def __init__(self):
        self.job_id = "999"
        self.pbar = tqdm(total=100)

    def update_message(self, message: Optional[str]) -> None:
        pass

    def update_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        progress = int(metadata["progress"] * 100)
        self.pbar.n = progress
        self.pbar.last_print_n = progress
        self.pbar.refresh()


class InstabaseSDKDummy:
    def __init__(self, file_client: Any, username: str):
        # these will be ignored
        self.file_client = file_client
        self.username = username

    def ibopen(self, path: str, mode: str = "r") -> Any:
        return open(path, mode)

    def read_file(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            return f.read()

    def write_file(self, file_path: str, content: str):
        # mkdir
        Path(file_path).parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "wb") as f:
            f.write(content)

    def unzip(self, file_path: str, destination: str, remove: bool = False):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(destination)
        if remove:
            os.remove(file_path)


def run(ds_path, out_path):
    hyperparams = {
        "batch_size": 4,
        "max_length": 512,
        "chunk_overlap": 16,
        "num_train_epochs": 1,
        "learning_rate": 5e-05,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_ratio": 0.0,
        "weight_decay": 0,
        "model_name": "microsoft/layoutlm-base-uncased",
        "pipeline_name": "layoutlm_cls",
        "task_type": "classification",
    }

    # ds_path = '/Users/rafalpowalski/python/annotation/UberEatsDataset'
    # out_path = '/Users/rafalpowalski/python/models/test_model'
    sdk = InstabaseSDKDummy(None, "user")
    run_train_both_classification(
        hyperparams=hyperparams,
        dataset_paths=[ds_path],
        save_path=out_path,
        file_client=sdk,
        username="user",
        job_metadata_client=DummyJobStatus(),
        mount_details=None,
        model_name="W2",
    )


if __name__ == "__main__":
    fire.Fire(run)
