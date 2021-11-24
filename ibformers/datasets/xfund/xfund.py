import json
import logging
import os

import datasets
import numpy as np
from PIL import Image

from ibformers.data.utils import ImageProcessor
from ibformers.datasets.utils import create_features_from_fund_file_content, enrich_features_with_images


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    # resize image to 224x224
    image = image.resize((224, 224))
    image = np.asarray(image)
    image = image[:, :, ::-1]  # flip color channels from RGB to BGR
    image = image.transpose(2, 0, 1)  # move channels to first dimension
    return image, (w, h)


def simplify_bbox(bbox):
    return [
        min(bbox[0::2]),
        min(bbox[1::2]),
        max(bbox[2::2]),
        max(bbox[3::2]),
    ]


def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]


_URL = "https://github.com/doc-analysis/XFUND/releases/download/v1.0/"
_DESCRIPTION = f"XFUNDS dataset: {_URL}"

_LANG = ["zh", "de", "es", "fr", "it", "ja", "pt"]
logger = logging.getLogger(__name__)


class XFUNConfig(datasets.BuilderConfig):
    """BuilderConfig for XFUN."""

    def __init__(self, use_image: bool = False, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(XFUNConfig, self).__init__(**kwargs)
        self.use_image = use_image
        self.lang = "de"
        self.additional_langs = "+".join(["zh", "es", "fr", "it", "ja", "pt"])  # TODO: do we want this configurable?


class XFUN(datasets.GeneratorBasedBuilder):
    """XFUN dataset."""

    BUILDER_CONFIGS = [XFUNConfig(name="xfund", version=datasets.Version("1.0.0"))]

    def __init__(self, *args, **kwargs):
        super(XFUN, self).__init__(*args, **kwargs)
        self.image_processor = ImageProcessor(do_resize=True, size=224) if self.config.use_image else None

    def _info(self):
        ds_features = {
            "id": datasets.Value("string"),
            "words": datasets.Sequence(datasets.Value("string")),
            "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
            "word_original_bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("float32"), length=4)),
            "word_page_nums": datasets.Sequence(datasets.Value("int32")),
            "page_bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=4)),
            "page_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
            "bio_token_label_ids": datasets.Sequence(
                datasets.features.ClassLabel(
                    names=["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
                )
            ),
            "token_label_ids": datasets.Sequence(
                datasets.features.ClassLabel(names=["O", "HEADER", "QUESTION", "ANSWER"])
            ),
            "entities": datasets.Sequence(
                {
                    "name": datasets.Value("string"),  # change to id?
                    "order_id": datasets.Value("int64"),
                    # not supported yet, annotation app need to implement it
                    "text": datasets.Value("string"),
                    "char_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
                    "token_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
                    "token_label_id": datasets.Value("int64"),
                }
            ),
        }
        if self.config.use_image:
            ds_features["images"] = datasets.Array4D(shape=(None, 3, 224, 224), dtype="uint8")
            ds_features["images_page_nums"] = datasets.Sequence(datasets.Value("int32"))

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(ds_features),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": [f"{_URL}{self.config.lang}.train.json", f"{_URL}{self.config.lang}.train.zip"],
            "val": [f"{_URL}{self.config.lang}.val.json", f"{_URL}{self.config.lang}.val.zip"],
            # "test": [f"{_URL}{self.config.lang}.test.json", f"{_URL}{self.config.lang}.test.zip"],
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        train_files_for_many_langs = [downloaded_files["train"]]
        val_files_for_many_langs = [downloaded_files["val"]]
        # test_files_for_many_langs = [downloaded_files["test"]]
        if self.config.additional_langs:
            additional_langs = self.config.additional_langs.split("+")
            if "all" in additional_langs:
                additional_langs = [lang for lang in _LANG if lang != self.config.lang]
            for lang in additional_langs:
                urls_to_download = {"train": [f"{_URL}{lang}.train.json", f"{_URL}{lang}.train.zip"]}
                additional_downloaded_files = dl_manager.download_and_extract(urls_to_download)
                train_files_for_many_langs.append(additional_downloaded_files["train"])

        logger.info(f"Training on {self.config.lang} with additional langs({self.config.additional_langs})")
        logger.info(f"Evaluating on {self.config.lang}")
        logger.info(f"Testing on {self.config.lang}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_files_for_many_langs}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": val_files_for_many_langs}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": val_files_for_many_langs}),
        ]

    def _generate_examples(self, filepaths):
        for ann_path, img_dir in filepaths:
            logger.info("Generating examples from = %s", (ann_path, img_dir))
            with open(ann_path, "r") as f:
                data = json.load(f)

            for doc in data["documents"]:
                guid = doc["id"]
                image_path = os.path.join(img_dir, doc["img"]["fname"])
                image, size = load_image(image_path)
                features = create_features_from_fund_file_content(
                    doc["document"], size, self.info.features["token_label_ids"].feature._str2int
                )
                features["id"] = guid
                if self.config.use_image:
                    enrich_features_with_images(features, image, self.image_processor)

                yield guid, features
