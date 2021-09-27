from __future__ import annotations

import logging
import os
import zipfile
from io import BytesIO


def zip_project(root_location: str) -> bytes:
    """
    Zips the project and returns the bytes of the zip file.
    """
    logging.info(f"Zipping project at {root_location}")
    bytesio: BytesIO = BytesIO()
    with zipfile.ZipFile(bytesio, "w") as zip_file:
        for root, dirs, files in os.walk(root_location):
            for file in files:
                if file in [".DS_Store", "__pycache__", "example"]: # TODO hardcoding example here is bad
                    continue
                logging.debug(f"Adding file {file} to zip")
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, root_location))
    return bytesio.getvalue()
