import logging
import os
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

    ibformers_do_not_copy = ['trainer']

    def __init__(self, job_status_client: 'JobStatusClient', ibsdk: InstabaseSDK,
                 username: str, mount_details: Dict, model_name: str, ib_save_path: str):
        self.ib_save_path = ib_save_path
        self.model_name = model_name
        self.mount_details = mount_details
        self.username = username
        self.job_status_client = job_status_client
        self.ibsdk = ibsdk

    def save_on_ib_storage(self, obj, filename):
        pass

    def build_package_structure(self, output_dir):

        package_name = self.model_name
        model_name = self.model_name
        model_class_name = self.model_name

        out_dir = Path(output_dir)
        # get ib_package location
        template_dir_path = _abspath('ib_package/ModelServiceTemplate')
        ibformers_path = Path(_abspath('')).parent
        save_folder = out_dir / 'saved_model'
        shutil.copytree(template_dir_path, save_folder)
        package_dir = save_folder / 'src' / 'py' / package_name
        shutil.move(package_dir.parent / 'package_name', package_dir)
        prepare_package_json(save_folder / "package.json",
                             model_name=model_name, model_class_name=model_class_name, package_name=package_name)

        # copy ibformers lib into the package, do not copy trainer
        shutil.copytree(ibformers_path, package_dir / 'ibformers', ignore=lambda x, y: self.ibformers_do_not_copy)

    def move_data_to_ib(self, output_dir):
        self.build_package_structure(output_dir)
        # copy data to ib



    def set_status(self, new_status: Dict):
        self.job_status_client.update_job_status(task_data=new_status)

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.set_status({"progress": state.global_step / state.max_steps})

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.set_status({"progress": state.global_step / state.max_steps})

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:

            if 'predict_predictions' in kwargs["metrics"]:
                predictions = kwargs["metrics"]['predict_predictions']
                self.move_data_to_ib(args.output_dir)
                a = 1

            else:
                metrics = {}
                metrics['precision'] = kwargs["metrics"]['eval_precision']
                metrics['recall'] = kwargs["metrics"]['eval_recall']
                metrics['f1'] = kwargs["metrics"]['eval_f1']
                self.set_status({"evaluation_results": metrics,
                                 "progress": state.global_step / state.max_steps})


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
        metadata={"help": "Where do you want to save ib_package on the IB space"},
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


def copy_from_template(model_name: str, model_class_name: str, package_name: str) -> str:
    """Creates a folder for the Model Service implementation of the
    LayoutLM model, without the model data. The model data should be
    inserted by the training script.
    Returns the root save folder of the model, as well as the location
    to save the model data specifically
    """

    # Create a folder for the location of the model
    tmp_folder = f'/tmp/data/{generate_randomness()}'
    save_folder = os.path.join(tmp_folder, 'saved_model')

    # Get the path to the template we want to populate
    template_dir_path = _abspath('res/ModelServiceTemplate')

    for root, _, files in os.walk(template_dir_path):
        for f in files:
            original_location = os.path.join(root, f)
            if root != template_dir_path:
                new_relative_location = os.path.relpath(root, template_dir_path)
                new_location = os.path.join(save_folder, new_relative_location, f)
            else:
                new_location = os.path.join(save_folder, f)

            new_location = new_location.replace('package_name', package_name)

            # For each file, read it, templatize, and write
            with open(original_location, 'r') as f_read:
                content = f_read.read()
                content = content.replace('{{model_package}}', package_name)
                content = content.replace('{{model_class_name}}', model_class_name)
                content = content.replace('{{model_name}}', model_name)

                if not os.path.exists(os.path.dirname(new_location)):
                    os.makedirs(os.path.dirname(new_location))

                with open(new_location, 'w+') as f_write:
                    f_write.write(content)

    model_data_location = os.path.join(save_folder, f'src/py/{package_name}/model_data')
    return tmp_folder, model_data_location
