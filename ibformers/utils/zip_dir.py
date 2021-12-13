import logging
import os
import zipfile
from pathlib import Path
from typing import List, Optional, Union


def zip_dir(
    root_location: Union[str, Path],
    output_location: Union[str, Path, zipfile.ZipFile],
    ignore_hidden: Optional[bool] = True,
    ignore_dirs: Optional[List] = None,
    ignore_files: Optional[List] = None,
) -> None:
    """
    Zips a directory to target location (could be a path or opened ZipFile object).
    :param: root_location: location of the directory to be zipped
    :param: output_location: location, to which directory is zipped, could be existing ZipFile object
    :param: ignore_hidden: whether to skip hidden files and directories when zipping
    :param: ignore_dirs: names or paths (full or relative) of directories to skip when zipping
    :param: ignore_files: names or paths (full or relative) of files to skip when zipping
    """
    zip_to_stream = False
    if isinstance(output_location, zipfile.ZipFile):
        zip_file = output_location
        zip_to_stream = True
    else:
        if not os.path.exists(os.path.dirname(output_location)):
            os.mkdir(os.path.dirname(output_location))
        zip_file = zipfile.ZipFile(output_location, "w")

    for dir_name, _, files in os.walk(root_location):
        # skip hidden directories
        if ignore_hidden and dir_name.startswith("."):
            logging.debug(f"Skipping dir {dir_name}")
            continue

        # skip marked directories
        if any([dir_name.endswith(dir_to_skip) for dir_to_skip in (ignore_dirs or [])]):
            logging.debug(f"Skipping dir {dir_name}")
            continue

        arcname_prefix = Path(os.path.basename(root_location))
        arcname_prefix /= Path(os.path.relpath(dir_name, root_location))
        for file_name in files:
            # skip hidden files
            if ignore_hidden and file_name.startswith("."):
                logging.debug(f"Skipping file {file_name}")
                continue

            # skip marked files
            if any([file_name.endswith(file_to_skip) for file_to_skip in (ignore_files or [])]):
                logging.debug(f"Skipping file {file_name}")
                continue

            file_path = os.path.join(dir_name, file_name)

            logging.debug(f"Adding file {file_name} to zip")
            zip_file.write(file_path, arcname=arcname_prefix / file_name)

    if not zip_to_stream:
        zip_file.close()
