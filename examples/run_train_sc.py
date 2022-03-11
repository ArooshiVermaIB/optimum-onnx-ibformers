from pathlib import Path
from typing import Optional, Dict, Any
from ibformers.trainer.docpro_utils import run_train_classification, run_train_both_classification
import fire
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


def run(ds_path, out_path):
    hyperparams = {
        "adam_epsilon": 1e-8,
        "batch_size": 2,
        "num_train_epochs": 2,
        "max_length": 512,
        "learning_rate": 5e-05,
        "gradient_accumulation_steps": 2,
        "max_grad_norm": 1.0,
        "optimizer_type": "AdamW",
        "lr_scheduler_type": "constant_schedule_with_warmup",
        "use_mixed_precision": False,
        "warmup_ratio": 0.0,
        "weight_decay": 0,
        "model_name": "microsoft/layoutlm-base-uncased",
        "pipeline_name": "layoutlm_sc",
        "task_type": "split_classification",
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
