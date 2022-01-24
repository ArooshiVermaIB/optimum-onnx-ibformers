from __future__ import annotations

import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path

TO_SKIP = ["__pycache__", ".DS_Store", ".gitignore"]


class PackageType(Enum):
    EXTRACTION: str = "ibformers_extraction"
    CLASSIFICATION: str = "ibformers_classification"


def zip_project(root_location: str, package_type: str) -> bytes:
    """
    Zips the project and returns the bytes of the zip file.
    """
    logging.info(f"Zipping project at {root_location}")
    bytesio: BytesIO = BytesIO()

    root_dir = os.path.basename(root_location)
    package_json_path = Path(root_location).parent / f"ci/package_config/{package_type}.json"
    target_package_json_path = Path(root_location).parent / "package.json"

    with zipfile.ZipFile(bytesio, "w") as zip_file:
        for root, dirs, files in os.walk(
            root_location,
        ):
            if any(f"/{i}/" in root or root.endswith(f"/{i}") for i in TO_SKIP):  # TODO Def a better way to do this
                logging.debug(f"Skipping dir {root}")
                continue
            for file in files:
                if file.startswith("."):
                    logging.debug(f"Skipping file {file}")
                    continue
                if file in TO_SKIP:
                    continue
                logging.debug(f"Adding file {file} to zip")
                file_path = os.path.join(root, file)
                zip_file.write(file_path, Path(root_dir) / os.path.relpath(file_path, root_location))

        logging.debug(f"Adding file package.json to zip")
        zip_file.write(package_json_path, os.path.relpath(target_package_json_path, Path(root_location).parent))

    return bytesio.getvalue()
