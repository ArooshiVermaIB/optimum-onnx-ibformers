import csv
from pathlib import Path

import fire
from PIL import Image, ImageSequence, UnidentifiedImageError


def get_number_of_pages_from_image(image_path: Path):
    try:
        im = Image.open(image_path)
    except UnidentifiedImageError:
        return 0
    i = 0
    for i, _ in enumerate(ImageSequence.Iterator(im)):
        pass
    return i + 1


def get_number_of_pages(doc_path: Path):
    if str(doc_path).endswith(".pdf"):
        raise NotImplementedError("Extracting the number of pages is not implemented yet!")
    return get_number_of_pages_from_image(doc_path)


def main(index_path: Path):
    file_paths = index_path.read_text().split("\n")
    page_numbers = [get_number_of_pages(Path(file_path)) for file_path in file_paths]

    index_dir, index_name = index_path.parent, index_path.name
    index_path.rename(index_dir / (index_name + ".old"))

    with index_path.open(mode="w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        for line in zip(file_paths, page_numbers):
            writer.writerow(line)


if __name__ == "__main__":
    fire.Fire(main)
