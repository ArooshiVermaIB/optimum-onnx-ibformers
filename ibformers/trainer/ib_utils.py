import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any, Iterable, Tuple
from typing_extensions import TypedDict
import boto3
import shutil

from transformers import TrainerCallback
from instabase.storage.fileservice import FileService
from instabase.content.filehandle import ibfile
from instabase.content.filehandle_lib.ibfile_lib import IBFileBase

logger = logging.getLogger(__name__)


class InstabaseSDK:
    def __init__(self, file_client: FileService.Iface, username: str):
        self.file_client = file_client
        self.username = username

    def ibopen(self, path: str, mode: str = 'r', **kwargs) -> IBFileBase:
        result = ibfile.ibopen(path, mode, file_client=self.file_client, username=self.username)
        return result

    def stat(self, path: str) -> FileService.StatResp:
        result, _ = ibfile.stat(path, file_client=self.file_client, username=self.username)
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


class IbCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that displays the progress of training or evaluation.
    it pass status of training to job_status_client and save required files to the IB location via ibsdk
    """

    ibformers_do_not_copy = []

    def __init__(
        self,
        job_status_client: 'JobStatusClient',
        ibsdk: InstabaseSDK,
        username: str,
        mount_details: Dict,
        model_name: str,
        ib_save_path: str,
        upload: bool,
    ):
        self.upload = upload
        self.ib_save_path = ib_save_path
        self.model_name = model_name
        self.mount_details = mount_details
        self.username = username
        self.job_status_client = job_status_client
        self.evaluation_results = None
        self.prediction_results = None
        self.ibsdk = ibsdk
        self.job_status = {}

    def save_on_ib_storage(self, obj, filename):
        pass

    def build_local_package_directory(self, output_dir):

        package_name = self.model_name
        model_name = self.model_name
        model_class_name = self.model_name

        out_dir = Path(output_dir)
        # get ib_package location
        template_dir_path = _abspath('ib_package/ModelServiceTemplate')
        ibformers_path = Path(_abspath('')).parent
        dir_to_be_copied = out_dir / 'package'
        save_model_dir = dir_to_be_copied / 'saved_model'
        shutil.copytree(template_dir_path, save_model_dir)
        package_dir = save_model_dir / 'src' / 'py' / package_name
        shutil.move(str(package_dir.parent / 'package_name'), str(package_dir))
        prepare_package_json(
            save_model_dir / "package.json",
            model_name=model_name,
            model_class_name=model_class_name,
            package_name=package_name,
        )

        # copy ibformers lib into the package
        # TODO will this have an issue without mkdir?
        shutil.copytree(
            ibformers_path,
            package_dir.parent / 'ibformers' / 'ibformers',
            ignore=lambda x, y: self.ibformers_do_not_copy,
        )

        # copy model files
        model_src_path = out_dir / 'model'
        assert model_src_path.is_dir(), 'Missing model files in output directory'
        model_dest_path = package_dir / 'model_data'
        if self.upload:
            for fl_path in model_src_path.iterdir():
                shutil.move(str(fl_path), str(model_dest_path))

        # save evaluation results
        if self.evaluation_results is not None:
            eval_path = dir_to_be_copied / 'evaluation.json'
            with open(eval_path, 'w') as f:
                json.dump(self.evaluation_results, f)

        # save prediction results
        if self.prediction_results is not None:
            pred_path = dir_to_be_copied / 'predictions.json'
            with open(pred_path, 'w') as f:
                json.dump(self.prediction_results, f)

        return dir_to_be_copied

    def move_data_to_ib(self, output_dir):
        dir_to_be_copied = self.build_local_package_directory(output_dir)
        # copy data to ib
        self.set_status({'task_state': 'UPLOADING FILES TO IB'})
        upload_dir(
            sdk=self.ibsdk,
            local_folder=dir_to_be_copied,
            remote_folder=self.ib_save_path,
            mount_details=self.mount_details,
        )
        self.set_status({'task_state': 'UPOLADING FINISHED'})

    def set_status(self, new_status: Dict):
        self.job_status.update(new_status)
        self.job_status_client.update_job_status(task_data=self.job_status)

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.set_status({"progress": state.global_step / state.max_steps})

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.set_status({"progress": state.global_step / state.max_steps})

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            # workaround for missing on_predict callback in the transformers TrainerCallback
            if 'predict_loss' in kwargs["metrics"]:
                self.on_predict(args, state, control, **kwargs)
            else:
                metrics = {}
                metrics['precision'] = kwargs["metrics"]['eval_precision']
                metrics['recall'] = kwargs["metrics"]['eval_recall']
                metrics['f1'] = kwargs["metrics"]['eval_f1']
                self.set_status(
                    {"evaluation_results": metrics, "progress": state.global_step / state.max_steps}
                )

                self.evaluation_results = metrics

    def on_predict(self, args, state, control, **kwargs):
        predictions = kwargs["metrics"]['predict_predictions']
        self.prediction_results = predictions
        # as prediction is the last step of the training - use this event to save the predictions to ib
        self.move_data_to_ib(args.output_dir)

        # This is a hacky way to let the frontend know that there are new preds available
        self.set_status({'predictions_uuid': uuid.uuid4().hex})


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
        default=None,
        metadata={
            "help": "Job status client. Used for collecting information of training progress"
        },
    )
    mount_details: Optional[Dict] = field(
        default=None,
        metadata={"help": "Store information about S3 details"},
    )
    model_name: Optional[str] = field(
        default="CustomModel",
        metadata={
            "help": "The model name which will be appear in the model management dashboard ??"
        },
    )
    ib_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to save ib_package on the IB space"},
    )
    upload: Optional[bool] = field(
        default=None, metadata={"help": "Whether to upload model files to ib_save_path"}
    )


def generate_randomness() -> str:
    """
    Get random five ascii letters
    """
    return ''.join(random.choice(string.ascii_letters) for _ in range(5))


class MountDetails(TypedDict):
    client_type: str  # Should be "S3" on prod
    prefix: str
    bucket_name: str
    aws_region: str


def prepare_package_json(path: str, model_name: str, model_class_name: str, package_name: str):
    # replace content with model details
    with open(path, 'r') as f_read:
        content = f_read.read()
        content = content.replace('{{model_package}}', package_name)
        content = content.replace('{{model_class_name}}', model_class_name)
        content = content.replace('{{model_name}}', model_name)

    with open(path, 'w+') as f_write:
        f_write.write(content)


def upload_dir(
    sdk: InstabaseSDK, local_folder: str, remote_folder: str, mount_details: Optional[MountDetails]
):
    s3 = get_s3_client() if mount_details and mount_details['client_type'] == "S3" else None
    logger.info(f"Uploading using " + ("S3" if s3 else "IB Filesystem"))
    for local, remote in map_directory_remote(local_folder, remote_folder):
        success = False
        if s3:
            success = s3_write(
                client=s3,
                local_file=local,
                remote_file=remote,
                bucket_name=mount_details['bucket_name'],
                prefix=mount_details['prefix'],
            )
            if not success:
                logging.debug(
                    "Upload with S3 was not successful. Falling back to using Instabase API."
                )
        if not s3 or not success:
            sdk.write_file(remote, open(local, 'rb').read())
        os.remove(local)
    logger.info("Finished uploading")


def map_directory_remote(local_folder, remote_folder) -> Iterable[Tuple[str, str]]:
    for folder_path, dirs, files in os.walk(local_folder):
        for f in files:
            local = os.path.join(folder_path, f)
            remote = os.path.join(remote_folder, os.path.relpath(local, local_folder))
            yield local, remote


def get_s3_client():
    # TODO: Invalidate these and replace with literally any better method
    return boto3.client(
        's3',
        aws_access_key_id="AKIARG3DSRG347TJWJXU",
        aws_secret_access_key="4RwcJffPHfNpA9bv4NuGYBeg5As4n3QJoTqg1e0w",
    )


def s3_write(client, local_file: str, remote_file: str, bucket_name: str, prefix: str) -> bool:
    # Returns True if successful; else False
    try:
        fs_index = remote_file.index('/fs/')
        start_index = remote_file.index('/', fs_index + 4)
        remote_file = remote_file[start_index:]
        client.upload_file(local_file, bucket_name, prefix + remote_file)
        return True
    except Exception as e:
        logging.debug(f"Error while uploading with S3: {repr(e)}")
        return False


def _abspath(relpath: str) -> str:
    dirpath, _ = os.path.split(__file__)
    return os.path.join(dirpath, relpath)


def prepare_ib_params(
    hyperparams: Dict,
    dataset_filename: str,
    save_path: str,
    file_client: Any,
    username: str,
    job_status_client: 'JobStatusClient',
    mount_details: Optional[Dict] = None,
    model_name: str = 'CustomModel',
) -> Dict:
    """
    Map parameters used by model service to names used in the Trainer
    :param hyperparams:
    :param dataset_filename:
    :param save_path:
    :param file_client:
    :param username:
    :param job_status_client:
    :param mount_details:
    :param model_name:
    :return:
    """
    hyperparams = {**hyperparams}

    temp_dir = tempfile.TemporaryDirectory().name
    out_dict = dict(
        do_train=True,
        do_eval=True,
        do_predict=True,
        log_level='warning',
        report_to='none',
        logging_strategy='epoch',
        evaluation_strategy='epoch',
        disable_tqdm=False,
        logging_steps=10,
        adafactor=False,
        dataset_name_or_path='ibds',
        dataset_config_name='ibds',
        train_file=dataset_filename,
        output_dir=temp_dir,
        ib_save_path=save_path,
        overwrite_output_dir=False,
        return_entity_level_metrics=True,
        username=username,
        file_client=file_client,
        job_status_client=job_status_client,
        mount_details=mount_details,
        model_name=model_name,
    )

    if 'epochs' in hyperparams:
        out_dict['num_train_epochs'] = hyperparams.pop('epochs')
    if 'batch_size' in hyperparams:
        out_dict['per_device_train_batch_size'] = int(hyperparams['batch_size'])
    if 'learning_rate' in hyperparams:
        out_dict['learning_rate'] = hyperparams.pop('learning_rate')
    if 'max_grad_norm' in hyperparams:
        out_dict['max_grad_norm'] = hyperparams.pop('max_grad_norm')
    if 'use_mixed_precision' in hyperparams:
        out_dict['fp16'] = hyperparams.pop('use_mixed_precision')
    if 'use_gpu' in hyperparams:
        out_dict['no_cuda'] = not hyperparams.pop('use_gpu')
    if 'warmup' in hyperparams:
        out_dict['warmup_ratio'] = hyperparams.pop('warmup')
    if 'weight_decay' in hyperparams:
        out_dict['weight_decay'] = hyperparams.pop('weight_decay')
    if 'chunk_size' in hyperparams:
        out_dict['max_length'] = int(hyperparams.pop('chunk_size'))
    if 'stride' in hyperparams:
        out_dict['chunk_overlap'] = int(hyperparams.pop('stride'))
    if 'upload' in hyperparams:
        out_dict['upload'] = hyperparams.pop('upload')

    if 'scheduler_type' in hyperparams:
        scheduler_type = hyperparams.pop('scheduler_type')
        if scheduler_type == "constant_schedule_with_warmup":
            out_dict['lr_scheduler_type'] = 'constant_with_warmup'
        elif scheduler_type == 'linear_schedule_with_warmup':
            out_dict['lr_scheduler_type'] = 'linear'
        else:
            out_dict['lr_scheduler_type'] = scheduler_type

    pipeline_name = None
    if 'pipeline_name' in hyperparams:
        pipeline_name = hyperparams.pop('pipeline_name')
    if 'model_name' in hyperparams:
        model_name = hyperparams.pop('model_name')
        out_dict['model_name_or_path'] = model_name
        if not pipeline_name:
            if 'layoutlmv2' in model_name.lower():
                pipeline_name = 'layoutlmv2_sl'
            elif 'layoutxlm' in model_name.lower():
                pipeline_name = 'layoutxlm_sl'
            else:
                pipeline_name = 'layoutlm_sl'
        out_dict['pipeline_name'] = pipeline_name

    if hyperparams:
        logging.info(
            f"The following hyperparams were ignored by the training loop: {hyperparams.keys()}"
        )

    return out_dict
