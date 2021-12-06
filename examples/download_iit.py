import logging
import os
import time
import urllib.request
from pathlib import Path

import fire
import httplib2
import requests
from bs4 import BeautifulSoup, SoupStrainer
from typing import List, Tuple

URL = "https://ir.nist.gov/cdip/cdip-images/"
NUM_FILES = 648


def get_file_list(url: str):
    logging.info(f"Getting file list at url {url}")
    http = httplib2.Http()
    status, response = http.request(url)
    all_links = []
    for link in BeautifulSoup(response, parse_only=SoupStrainer("a")):
        if link.has_attr("href") and "tar" in link["href"]:
            all_links.append("/".join((url, link["href"])))
    return all_links


def get_unfinished_tasks(file_list: List[str], download_dir: Path) -> List[Tuple[str, Path]]:
    all_tasks = [(file_url, download_dir / file_url.split("/")[-1]) for file_url in file_list]
    assert len(all_tasks) == NUM_FILES
    unfinished_tasks = [(url, path) for (url, path) in all_tasks if not verify_file(url, path)]
    logging.info(f"Found {len(unfinished_tasks)} files to download.")
    return unfinished_tasks


def get_file_size_from_server(url: str) -> int:
    logging.info(f"Getting file size at url {url}")
    r = requests.get(str(url), stream=True)
    return int(r.headers["content-length"])


def get_file_size_from_disk(path: Path) -> int:
    logging.info(f"Getting file size at path {path}")
    return os.stat(path).st_size


def verify_file(url: str, download_path: Path) -> bool:
    logging.info(f"Verifying {url} in file {download_path}")
    if not download_path.exists():
        return False
    url_size = get_file_size_from_server(url)
    downloaded_size = get_file_size_from_disk(download_path)
    return url_size == downloaded_size


def get_file(url: str, download_path: Path) -> None:
    logging.info(f"Downloading from {url} to {download_path} file")
    urllib.request.urlretrieve(url, download_path)


def get_and_verify(url: str, download_path: Path, retry_count: int, sleep_time: int) -> bool:
    logging.info(f"Downloading and verifying {url} to {download_path} file.")
    counter = 0
    success = False
    while counter < retry_count:
        try:
            get_file(url, download_path)
        except Exception as e:
            counter += 1
            logging.warning(f"Encountered exception when downloading {url}. Full message: {e}. Retrying: {counter}")
            time.sleep(sleep_time)
            continue
        success = verify_file(url, download_path)
        if success:
            break
        else:
            counter += 1
            logging.warning(f"Veryfication failed when downloading {url}.")
            time.sleep(sleep_time)
            continue
    if not success:
        logging.warning(f"Downloading {url} failed {retry_count} times. Skipping file.")
    return success


def main(url: str = URL, download_dir: str = "downloads", retry_count: int = 3, sleep_time: int = 10):
    download_dir = Path(download_dir)
    download_dir.mkdir(exist_ok=True)
    file_list = get_file_list(url)
    tasks_to_download = get_unfinished_tasks(file_list, download_dir)
    for file_url, file_path in tasks_to_download:
        get_and_verify(file_url, file_path, retry_count, sleep_time)


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
    )
    fire.Fire(main)
