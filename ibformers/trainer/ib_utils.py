from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from transformers import TrainerCallback
from instabase.storage.fileservice import FileService
from instabase.content.filehandle import ibfile
from instabase.content.filehandle_lib.ibfile_lib import IBFileBase


class InstabaseSDK:
    def __init__(self, file_client: FileService.Iface, username: str):
        self.file_client = file_client
        self.username = username

    # def __dict__(self):
    #     return {"file_client": str(self.file_client),
    #             "username": self.username}
    #
    # def __getstate__(self):
    #     return {"file_client": str(self.file_client),
    #             "username": self.username}

    def ibopen(self, path: str, mode: str = 'r', **kwargs) -> IBFileBase:
        result = ibfile.ibopen(path, mode, file_client=self.file_client, username=self.username)
        return result

    def stat(self, path: str) -> FileService.StatResp:
        result, _ = ibfile.stat(path, file_client=self.file_client, username=self.username)
        print(result)
        return result

    def read_file(self, file_path: str) -> str:
        result, error = ibfile.read_file(
            file_path, username=self.username, file_client=self.file_client
        )
        if error:
            raise IOError(error)
        return result

    def write_file(self, file_path: str, content):
        result, error = ibfile.write_file(
            file_path, content, username=self.username, file_client=self.file_client
        )
        if error:
            raise IOError(error)
        return result


class IbLogCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that displays the progress of training or evaluation.
    """

    def __init__(self, job_status_client: 'JobStatusClient'):
        self.job_status_client = job_status_client

    def set_status(self, new_status: Dict):
        self.job_status_client.update_job_status(task_data=new_status)

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.set_status({"progress": state.global_step / state.max_steps})

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            metrics = {}
            metrics['precision'] = kwargs["metrics"]['eval_precision']
            metrics['recall'] = kwargs["metrics"]['eval_recall']
            metrics['f1'] = kwargs["metrics"]['eval_f1']
            self.set_status({"evaluation_results": metrics,
                             "progress": state.global_step / state.max_steps})

            # generate predictions & print it to the logger
            assert 'eval_dataloader' in kwargs, 'Eval dataloder is missing in order to generate predictions'






@dataclass
class IbArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    username: Optional[str] = field(
        metadata={"help": "Username of person who is running the model training"}
    )
    file_client: Optional[Any] = field(
        default=None, metadata={"help": "File client object which support different file systems"}
    )
    job_status_client: Optional['JobStatusClient'] = field(
        default=None, metadata={"help": "Job status client. Used for collecting information of training progress"}
    )
    mount_details: Optional[Dict] = field(
        default=None,
        metadata={"help": "Store information about S3 details"},
    )
    model_name: Optional[str] = field(
        default="CustomModel",
        metadata={"help": "The model name which will be appear in the model management dashboard ??"},
    )
    ib_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to save package on the IB space"},
    )
