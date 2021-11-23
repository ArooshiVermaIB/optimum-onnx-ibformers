# coding=utf-8
import json
import os
from collections import defaultdict

import datasets
from PIL import Image
import numpy as np
from typing import Any, Dict, Tuple

from ibformers.data.utils import ImageProcessor

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


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


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

    def _create_features_from_file_content(
        self, file_content: Dict[str, Any], image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        label2id = self.info.features["token_label_ids"].feature._str2int

        features = defaultdict(list)
        features['page_bboxes'].append([0, 0, *image_size])

        entity_dicts = {
            label: {
                'name': label,
                'order_id': 0,
                'text': '',
                'char_spans': [],
                'token_label_id': label2id[label],
                'token_spans': [],
            }
            for label in label2id.keys()
            if label != 'O'
        }

        word_id_counter = 0
        for item in file_content["form"]:
            words_example, label = item["words"], item["label"]
            label = label.upper()
            words_example = [w for w in words_example if w["text"].strip() != ""]
            if len(words_example) == 0:
                continue
            if label == "OTHER":
                for w in words_example:
                    features['words'].append(w["text"])
                    features['token_label_ids'].append("O")
                    features['bio_token_label_ids'].append("O")
                    features['bboxes'].append(normalize_bbox(w["box"], image_size))
                    features['word_original_bboxes'].append(w["box"])
                    features['word_page_nums'].append(0)
                    word_id_counter += 1
            else:
                entity_dict = entity_dicts[label]
                entity_text = ' '.join([w['text'] for w in words_example])
                start_word_id = word_id_counter
                features['words'].append(words_example[0]["text"])
                features['token_label_ids'].append(label)
                features['bio_token_label_ids'].append("B-" + label)
                features['bboxes'].append(normalize_bbox(words_example[0]["box"], image_size))
                features['word_original_bboxes'].append(words_example[0]["box"])
                features['word_page_nums'].append(0)
                word_id_counter += 1
                for w in words_example[1:]:
                    features['words'].append(w["text"])
                    features['token_label_ids'].append(label)
                    features['bio_token_label_ids'].append("I-" + label)
                    features['bboxes'].append(normalize_bbox(w["box"], image_size))
                    features['word_original_bboxes'].append(w["box"])
                    features['word_page_nums'].append(0)
                    word_id_counter += 1
                entity_dict['token_spans'].append([start_word_id, word_id_counter])
                entity_dict['text'] += f' {entity_text}'
        features['entities'] = list(entity_dicts.values())
        return features

    def _enrich_features_with_images(self, features, image):
        word_pages_arr = features['word_page_nums']
        page_nums = np.unique(word_pages_arr)
        page_nums.sort()
        image_postprocessed = self.image_processor.postprocess(image)
        # assert len(norm_page_bboxes) == len(images), "Number of images should match number of pages in document"
        features["images"] = [image_postprocessed]
        features["images_page_nums"] = page_nums

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
            features = self._create_features_from_file_content(data, size)
            features['id'] = guid
            if self.config.use_image:
                self._enrich_features_with_images(features, image)

            yield guid, features
