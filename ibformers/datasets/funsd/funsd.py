# coding=utf-8
import json
import os
from collections import defaultdict

import datasets
from PIL import Image
import numpy as np
from typing import Any, Dict, Tuple

from ibformers.data.utils import ImageProcessor
from ibformers.datasets.utils import create_features_from_file_content, enrich_features_with_images

logger = datasets.logging.get_logger(__name__)
_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""
_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, use_image: bool = False, ibsdk=None, **kwargs):
        """BuilderConfig for FUNSD.
        Args:
          use_image: if true, the images are added as features
          ibsdk: param to allow config initialization. Not actually required.
          **kwargs: keyword arguments forwarded to super.
        """
        super(FunsdConfig, self).__init__(**kwargs)
        self.use_image = use_image
        self.ibsdk = ibsdk


class Funsd(datasets.GeneratorBasedBuilder):
    """FUNSD dataset."""

    def __init__(self, *args, **kwargs):
        super(Funsd, self).__init__(*args, **kwargs)
        self.image_processor = ImageProcessor(do_resize=True, size=224) if self.config.use_image else None

    BUILDER_CONFIGS = [
        FunsdConfig(name="funsd", version=datasets.Version("1.0.0"), description="FUNSD dataset"),
    ]

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
            homepage="https://guillaumejaume.github.io/FUNSD/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/training_data/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testing_data/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testing_data/"}
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
            features = create_features_from_file_content(
                data["form"], size, self.info.features["token_label_ids"].feature._str2int
            )
            features["id"] = guid
            if self.config.use_image:
                enrich_features_with_images(features, image, self.image_processor)

            yield guid, features
