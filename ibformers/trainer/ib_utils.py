import json
import logging
import math
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Optional, Any, Iterable, Tuple
from typing_extensions import TypedDict
import boto3
import shutil
import zipfile

from transformers import TrainerCallback, HfArgumentParser

from ibformers.data.collators.augmenters.args import AugmenterArguments
from ibformers.trainer.train import run_train
from ibformers.trainer.arguments import (
    ModelArguments,
    DataAndPipelineArguments,
    IbArguments,
    update_params_with_commandline,
    EnhancedTrainingArguments,
)

from instabase.storage.fileservice import FileService
from instabase.content.filehandle import ibfile
from instabase.content.filehandle_lib.ibfile_lib import IBFileBase, default_max_write_size
from instabase.utils.rpc.file_client import ThriftGRPCFileClient

# imports for unzipping functionality
from instabase.utils.concurrency import executors
from instabase.utils.concurrency.types import LocalThreadConfig
from instabase.utils.path.ibpath import join
from instabase.utils.strutils import decode


logger = logging.getLogger(__name__)


_IBFILE_CHUNK_SIZE_IN_MB = 500
_PARALLEL_UPLOAD_WORKERS = 5

# this function is copied from instabase/tasks/file_fns.py
def _extract_item(zip_file_ref, filename_raw, save_location, file_client, username):
    # type: (zipfile.ZipFile, str, Text, FileService.Iface, Text) -> Text
    """Given a ZipFile reference, file to extract, and save location for that
    extraction, extracts that file into the correct location.
    """
    if True:  # PY3:
        # See: https://stackoverflow.com/questions/41019624/python-zipfile-module-cant-extract-filenames-with-chinese-characters
        # NOTE: In Python 3, the zipfile library makes some very specific
        # assumptions about filename encoding. ZIP files may include a flag that
        # guarantees that a file name is encoded in UTF-8. If this flag is not set,
        # zipfile assumes that the name is encoded in cp437. In practice, however,
        # this assumption is often wrong, which can result in incorrect file names.

        # For documentation on the flag as part of the ZIP specification,
        # see appendix D of https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
        ZIP_UTF_FLAG = 1 << 11

        zip_info = zip_file_ref.getinfo(filename_raw)
        if zip_info.flag_bits & ZIP_UTF_FLAG:
            # filename is encoded with UTF-8
            filename = filename_raw
        else:
            # zipfile will have assumed that filename is encoded with cp437.
            # In general, there's no way of telling what encoding was actually used,
            # but UTF-8 is probably a good enough guess most of the time.
            filename = filename_raw.encode("cp437").decode("utf-8")
    else:
        filename = decode(filename_raw)

    # If this is an empty folder, create the folder
    if filename.endswith("/"):
        directory = join(save_location, filename)
        if not ibfile.exists(directory, file_client, username):
            ibfile.mkdir(directory, file_client, username, create_dirs=True)
        return None

    # Otherwise, create the new file (and its directory if needed)
    content = zip_file_ref.read(filename_raw)
    ibfile.write_file(join(save_location, filename), content, username=username, file_client=file_client)

    return None


class InstabaseSDK:
    def __init__(self, file_client: FileService.Iface, username: str, use_write_multipart: bool = True):
        self.file_client = file_client
        self.username = username
        self._CHUNK_SIZE = min(default_max_write_size, _IBFILE_CHUNK_SIZE_IN_MB * 1024 * 1024)

        self.use_multipart = use_write_multipart and isinstance(file_client, ThriftGRPCFileClient)

    def ibopen(self, path: str, mode: str = "r") -> IBFileBase:
        result = ibfile.ibopen(path, mode, file_client=self.file_client, username=self.username)
        return result

    def read_file(self, file_path: str) -> bytes:
        result, error = ibfile.read_file(file_path, username=self.username, file_client=self.file_client)
        if error:
            raise IOError(error)
        return result

    def _chunked_write_file(self, file_path: str, content: bytes):
        chunk_size = self._CHUNK_SIZE
        total_chunks = math.ceil(len(content) / chunk_size)

        with self.ibopen(file_path, "wb") as f:
            for i, chunk in enumerate(range(0, len(content), chunk_size)):
                chunk_bytes = content[chunk : chunk + chunk_size]
                cur_chunk_size_mb = len(chunk_bytes) / 1024 / 1024
                logging.info(
                    f"[InstabaseSDK._chunked_write_file] Writing chunk# {i+1}/{total_chunks}, {cur_chunk_size_mb:.3f} MB"
                )
                f.write(chunk_bytes)  # raises Exception

    def write_file(self, file_path: str, content: bytes):
        if self.use_multipart and (len(content) > 0):
            logging.info(f"[InstabaseSDK.write_file] Uploading {file_path} with multipart.")
            ibfile.write_file_multipart(
                file_path,
                content,
                username=self.username,
                file_client=self.file_client,
                max_workers=_PARALLEL_UPLOAD_WORKERS,
            )
        else:
            logging.info(f"[InstabaseSDK.write_file] Uploading {file_path} with chunked write.")
            self._chunked_write_file(file_path, content)

    # copied from instabase/tasks/file_fns with minimal changes
    def unzip(self, file_path: str, destination: str, remove: bool = False):
        """
        Unzip folder on the IB file system

        :param: file_path: IB path of a zip file
        :param: destination: IB path, where zip file's content will be extracted into
        :param: remove: whether to remove original zip file after extraction
        """
        # First check for existence
        if not ibfile.exists(file_path, file_client=self.file_client, username=self.username):
            raise IOError(f"Zip file {file_path} does not exist")
        if not ibfile.is_file(file_path, file_client=self.file_client, username=self.username):
            raise IOError(f"Zip path {file_path} is not a valid file")

        # Iterate through zip and save files. Wrap the entire operation in a
        # try catch (for instance, if this is not a valid zip file).
        destination = destination.rstrip("/")

        try:
            with ibfile.ibopen(file_path, "rb", file_client=self.file_client, username=self.username) as f:
                # ibfile should take care of chunking file contents if necessary
                with zipfile.ZipFile(f, "r") as zf:
                    # For each file, extract that file into the save location
                    config = LocalThreadConfig(max_workers=10)
                    with executors.NewExecutor(config) as executor:
                        extract_jobs = [
                            executor.submit(_extract_item, zf, filename, destination, self.file_client, self.username)
                            for filename in zf.namelist()
                        ]
                        executors.wait(extract_jobs, timeout=1200)
                        for job in extract_jobs:
                            err_msg = job.value()
                            if job.exception():
                                if err_msg == None:
                                    err_msg = ""
                                err_msg += f" Job exception: {str(job.exception())}"
                            if err_msg:
                                raise Exception("An error occurred while extracting this ZIP file: " + err_msg)
            if remove:
                ibfile.rm(file_path, file_client=self.file_client, username=self.username)
        except Exception as e:
            raise Exception(f"An error occurred while extracting this ZIP file: {e}")


class IbCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that displays the progress of training or evaluation.
    it pass status of training to job_metadata_client and save required files to the IB location via ibsdk
    """

    ibformers_do_not_copy = []

    def __init__(
        self,
        job_metadata_client: "JobMetadataClient",
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
        self.job_metadata_client = job_metadata_client
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
        template_dir_path = _abspath("ib_package/ModelServiceTemplate")
        ibformers_path = Path(_abspath("")).parent
        dir_to_be_copied = out_dir / "package"
        save_model_dir = dir_to_be_copied / "saved_model"
        shutil.copytree(template_dir_path, save_model_dir)
        package_dir = save_model_dir / "src" / "py" / package_name
        shutil.move(str(package_dir.parent / "package_name"), str(package_dir))
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
            package_dir.parent / "ibformers" / "ibformers",
            ignore=lambda x, y: self.ibformers_do_not_copy,
        )

        # copy model files
        model_src_path = out_dir / "model"
        assert model_src_path.is_dir(), "Missing model files in output directory"
        model_dest_path = package_dir / "model_data"
        if self.upload:
            for fl_path in model_src_path.iterdir():
                shutil.move(str(fl_path), str(model_dest_path))

        # save evaluation results
        if self.evaluation_results is not None:
            eval_path = dir_to_be_copied / "evaluation.json"
            with open(eval_path, "w") as f:
                json.dump(self.evaluation_results, f)

        # save prediction results
        if self.prediction_results is not None:
            pred_path = dir_to_be_copied / "predictions.json"
            with open(pred_path, "w") as f:
                json.dump(self.prediction_results, f)

        return dir_to_be_copied

    def move_data_to_ib(self, output_dir):
        dir_to_be_copied = self.build_local_package_directory(output_dir)
        # copy data to ib
        self.set_status({"task_state": "UPLOADING FILES TO IB"})
        upload_dir(
            sdk=self.ibsdk,
            local_folder=dir_to_be_copied,
            remote_folder=self.ib_save_path,
            mount_details=self.mount_details,
        )
        self.set_status({"task_state": "UPOLADING FINISHED"})

    def set_status(self, new_status: Dict):
        self.job_status.update(new_status)
        self.job_metadata_client.update_metadata(self.job_status)

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.set_status({"progress": state.global_step / state.max_steps})

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.set_status({"progress": state.global_step / state.max_steps})

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            # workaround for missing on_predict callback in the transformers TrainerCallback
            if "predict_loss" in kwargs["metrics"]:
                self.on_predict(args, state, control, **kwargs)
            elif "eval_loss" in kwargs["metrics"]:
                metrics = {}
                metrics["precision"] = kwargs["metrics"]["eval_precision"]
                metrics["recall"] = kwargs["metrics"]["eval_recall"]
                metrics["f1"] = kwargs["metrics"]["eval_f1"]
                self.set_status({"evaluation_results": metrics, "progress": state.global_step / state.max_steps})

                self.evaluation_results = metrics
            else:
                # ignore last evaluation call
                pass

    def on_predict(self, args, state, control, **kwargs):
        predictions = kwargs["metrics"]["predict_predictions"]
        self.prediction_results = predictions
        # as prediction is the last step of the training - use this event to save the predictions to ib
        self.move_data_to_ib(args.output_dir)

        # This is a hacky way to let the frontend know that there are new preds available
        self.set_status({"predictions_uuid": uuid.uuid4().hex})


class MountDetails(TypedDict):
    client_type: str  # Should be "S3" on prod
    prefix: str
    bucket_name: str
    aws_region: str


def prepare_package_json(path: str, model_name: str, model_class_name: str, package_name: str):
    # replace content with model details
    with open(path, "r") as f_read:
        content = f_read.read()
        content = content.replace("{{model_package}}", package_name)
        content = content.replace("{{model_class_name}}", model_class_name)
        content = content.replace("{{model_name}}", model_name)

    with open(path, "w+") as f_write:
        f_write.write(content)


def upload_dir(sdk: InstabaseSDK, local_folder: str, remote_folder: str, mount_details: Optional[MountDetails]):
    s3 = get_s3_client() if mount_details and mount_details["client_type"] == "S3" else None
    logging.info(f"Uploading using " + ("S3" if s3 else "IB Filesystem"))
    for local, remote in map_directory_remote(local_folder, remote_folder):
        success = False
        if s3:
            success = s3_write(
                client=s3,
                local_file=local,
                remote_file=remote,
                bucket_name=mount_details["bucket_name"],
                prefix=mount_details["prefix"],
            )
            if not success:
                logging.warning("Upload with S3 was not successful. Falling back to using Instabase API.")
        if not s3 or not success:
            with open(local, "rb") as f:
                sdk.write_file(remote, f.read())
        os.remove(local)
    logging.info("Finished uploading")


def map_directory_remote(local_folder, remote_folder) -> Iterable[Tuple[str, str]]:
    for folder_path, dirs, files in os.walk(local_folder):
        for f in files:
            local = os.path.join(folder_path, f)
            remote = os.path.join(remote_folder, os.path.relpath(local, local_folder))
            yield local, remote


def get_s3_client() -> Optional[boto3.session.Session.client]:
    aws_access_key_id = os.environ.get("aws_access_key_id", None)
    aws_secret_access_key = os.environ.get("aws_secret_access_key", None)

    if aws_access_key_id and aws_secret_access_key:
        return boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    logging.warning("[get_s3_client] AWS credentials not found in environment!")
    return None


def s3_write(client, local_file: str, remote_file: str, bucket_name: str, prefix: str) -> bool:
    # Returns True if successful; else False
    try:
        fs_index = remote_file.index("/fs/")
        start_index = remote_file.index("/", fs_index + 4)
        remote_file = remote_file[start_index:]
        client.upload_file(local_file, bucket_name, prefix + remote_file)
        return True
    except Exception as e:
        logging.warning(f"[s3_write] {repr(e)}")
        return False


def _abspath(relpath: str) -> str:
    dirpath, _ = os.path.split(__file__)
    return os.path.join(dirpath, relpath)


def prepare_ib_params(
    hyperparams: Dict,
    dataset_filename: Optional[str],
    save_path: str,
    file_client: Any,
    username: str,
    job_metadata_client: "JobMetadataClient",
    mount_details: Optional[Dict] = None,
    model_name: str = "CustomModel",
) -> Dict:
    """
    Map parameters used by model service to names used in the Trainer
    :param hyperparams:
    :param dataset_filename:
    :param save_path:
    :param file_client:
    :param username:
    :param job_metadata_client:
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
        log_level="info",
        report_to="none",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="no",
        disable_tqdm=False,
        logging_steps=10,
        adafactor=False,
        dataset_name_or_path="ibds",
        dataset_config_name="ibds",
        train_file=dataset_filename,
        output_dir=temp_dir,
        ib_save_path=save_path,
        overwrite_output_dir=False,
        return_entity_level_metrics=True,
        username=username,
        file_client=file_client,
        job_metadata_client=job_metadata_client,
        mount_details=mount_details,
        model_name=model_name,
        final_model_dir=os.path.join(temp_dir, "model"),
    )

    if "epochs" in hyperparams:
        out_dict["num_train_epochs"] = hyperparams.pop("epochs")
    if "batch_size" in hyperparams:
        out_dict["per_device_train_batch_size"] = int(hyperparams.pop("batch_size"))
    if "gradient_accumulation_steps" in hyperparams:
        out_dict["gradient_accumulation_steps"] = int(hyperparams.pop("gradient_accumulation_steps"))
    if "learning_rate" in hyperparams:
        out_dict["learning_rate"] = hyperparams.pop("learning_rate")
    if "max_grad_norm" in hyperparams:
        out_dict["max_grad_norm"] = hyperparams.pop("max_grad_norm")
    if "use_mixed_precision" in hyperparams:
        out_dict["fp16"] = hyperparams.pop("use_mixed_precision")
    if "use_gpu" in hyperparams:
        out_dict["no_cuda"] = not hyperparams.pop("use_gpu")
    if "warmup" in hyperparams:
        out_dict["warmup_ratio"] = hyperparams.pop("warmup")
    if "weight_decay" in hyperparams:
        out_dict["weight_decay"] = hyperparams.pop("weight_decay")
    if "chunk_size" in hyperparams:
        out_dict["max_length"] = int(hyperparams.pop("chunk_size"))
    if "stride" in hyperparams:
        out_dict["chunk_overlap"] = int(hyperparams.pop("stride"))
    if "upload" in hyperparams:
        out_dict["upload"] = hyperparams.pop("upload")

    if "scheduler_type" in hyperparams:
        scheduler_type = hyperparams.pop("scheduler_type")
        if scheduler_type == "constant_schedule_with_warmup":
            out_dict["lr_scheduler_type"] = "constant_with_warmup"
        elif scheduler_type == "linear_schedule_with_warmup":
            out_dict["lr_scheduler_type"] = "linear"
        else:
            out_dict["lr_scheduler_type"] = scheduler_type

    pipeline_name = None
    if "pipeline_name" in hyperparams:
        pipeline_name = hyperparams.pop("pipeline_name")
    if "model_name" in hyperparams:
        model_name = hyperparams.pop("model_name")
        out_dict["model_name_or_path"] = model_name
        if not pipeline_name:
            if "layoutlmv2" in model_name.lower():
                pipeline_name = "layoutlmv2_sl"
            elif "layoutxlm" in model_name.lower():
                pipeline_name = "layoutxlm_sl"
            elif "laymqav1" in model_name.lower():
                pipeline_name = "laymqav1"
            else:
                pipeline_name = "layoutlm_sl"
        out_dict["pipeline_name"] = pipeline_name
    if "report_to" in hyperparams:
        out_dict["report_to"] = hyperparams.pop("report_to")

    if "class_weights" in hyperparams:
        out_dict["class_weights"] = hyperparams.pop("class_weights")

    if hyperparams:
        logging.warning(f"The following hyperparams were ignored by the training loop: {hyperparams.keys()}")

    return out_dict


def run_train_annotator(
    hyperparams: Dict,
    dataset_filename: str,
    save_path: str,
    file_client: Any,
    username: Optional[str] = "user",
    job_metadata_client: Optional["JobMetadataClient"] = None,
    mount_details: Optional[Dict] = None,
    model_name: Optional[str] = "CustomModel",
    overwrite_arguments_with_cli: bool = False,
    **kwargs: Any,
):
    assert hyperparams is not None
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataAndPipelineArguments,
            EnhancedTrainingArguments,
            IbArguments,
            AugmenterArguments,
        )
    )

    hparams_dict = prepare_ib_params(
        hyperparams,
        dataset_filename,
        save_path,
        file_client,
        username,
        job_metadata_client,
        mount_details,
        model_name,
    )
    model_args, data_args, training_args, ib_args, augmenter_args = parser.parse_dict(hparams_dict)
    if overwrite_arguments_with_cli:
        model_args, data_args, training_args, ib_args, augmenter_args = update_params_with_commandline(
            (model_args, data_args, training_args, ib_args, augmenter_args)
        )

    if hasattr(file_client, "file_client") and file_client.file_client is None:
        # support for InstabaseSDKDummy - debugging only
        ibsdk = file_client
    else:
        ibsdk = InstabaseSDK(file_client, username)

    callback = IbCallback(
        job_metadata_client=ib_args.job_metadata_client,
        ibsdk=ibsdk,
        username=ib_args.username,
        mount_details=ib_args.mount_details,
        model_name=ib_args.model_name,
        ib_save_path=ib_args.ib_save_path,
        upload=ib_args.upload,
    )

    run_train(
        model_args,
        data_args,
        training_args,
        ib_args,
        augmenter_args,
        extra_callbacks=[callback],
        extra_load_kwargs={"ibsdk": ibsdk},
    )


class DummyJobStatus:
    def __init__(self):
        pass

    def update_message(self, message: Optional[str]) -> None:
        pass

    def update_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        pass

    def job_id(self) -> str:
        pass

    def get_job_status(self):
        pass

    def set_job_status(self, metadata) -> None:
        pass

    def update_job_status(self, **updates: Any) -> None:
        pass
