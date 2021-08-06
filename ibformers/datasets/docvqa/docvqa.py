import json
from typing import List, Tuple

import datasets
from datasets.tasks import QuestionAnsweringExtractive
from pathlib import Path
import cv2
from dataclasses import dataclass

_DESCRIPTION = """\
The objective of this task is to answer questions asked on a document image. The images provided are sourced from 
the documents hosted at the Industry Documents Library, maintained by the UCSF. The documents contain a mix of printed, 
typewritten and handwritten content. A wide variety of document types is used for this task 
including letters, memos, notes, reports etc.
The answers to questions are short text spans taken verbatim from the document. 
This means that the answers comprise a set of contiguous text tokens present in the document.
"""


BoundingBox = Tuple[float, float, float, float]
Span = Tuple[int, int]


@dataclass()
class OcrDoc:
    doc_id: str
    words: List[str]
    bboxes: List[BoundingBox]
    page_bboxes: List[BoundingBox]
    page_spans: List[Span]


def norm_bbox(bbox, width):
    # divide only by width to keep information about ratio
    return (
        bbox[0] / width,
        bbox[1] / width,
        bbox[2] / width,
        bbox[3] / width,
    )


def load_msocr_file(ocr_path):
    with open(ocr_path, encoding="utf-8") as f:
        ocr_json = json.load(f)

        example = {}
        example["ocr_path"] = ocr_path
        example["qas"] = []
        words = []
        bboxes = []
        bboxes_norm = []
        line_indices = []
        lines_array = []
        page_bboxes = []
        page_spans = []
        prev_page_total_words = 0

        # Added boxes and context to the example
        for obj in ocr_json['recognitionResults']:
            width, height = obj["width"], obj["height"]
            lines = obj['lines']
            idx = 0
            for line in lines:
                lines_array.append(line['text'])
                for word in line['words']:
                    words.append(word['text'])
                    line_indices.append(idx)
                    x1, y1, x2, y2, x3, y3, x4, y4 = word['boundingBox']
                    new_x1 = min([x1, x2, x3, x4])
                    new_x2 = max([x1, x2, x3, x4])
                    new_y1 = min([y1, y2, y3, y4])
                    new_y2 = max([y1, y2, y3, y4])
                    bboxes.append([new_x1, new_y1, new_x2, new_y2])
                    box_norm = norm_bbox([new_x1, new_y1, new_x2, new_y2], width)
                    assert new_x2 >= new_x1
                    assert new_y2 >= new_y1
                    assert box_norm[2] >= box_norm[0]
                    assert box_norm[3] >= box_norm[1]

                    bboxes_norm.append(box_norm)
                    idx += 1

            page_bboxes.append(norm_bbox([0, 0, width, height], width))
            page_spans.append((prev_page_total_words, len(words)))
            prev_page_total_words = len(words)

        assert len(words) == len(bboxes_norm)

    doc = OcrDoc(doc_id=str(ocr_path), words=words, bboxes=bboxes_norm, page_bboxes=page_bboxes, page_spans=page_spans)

    return doc


class DocvqaConfig(datasets.BuilderConfig):
    """
    Config for Docvqa Datasets
    """

    def __init__(self, use_image=False, **kwargs):
        """BuilderConfig for Docvqa.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DocvqaConfig, self).__init__(**kwargs)
        self.use_image = use_image


class Docvqa(datasets.GeneratorBasedBuilder):
    """TODO(docvqa): Short description of my dataset."""

    BUILDER_CONFIGS = [
        DocvqaConfig(name="docvqa", version=datasets.Version("1.0.0"), description="DocVQA dataset"),
    ]

    def _info(self):
        # TODO(squad_v2): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "image_id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("float"), length=4)),
                    "page_bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("float"), length=4)),
                    "page_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int64"), length=2)),
                    "question": datasets.Value("string"),
                    "anwers": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: set up a s3 bucket and upload the files there
        # download and extract URLs
        # urls_to_download = _URLS
        # downloaded_files = dl_manager.download_and_extract(urls_to_download)

        # requirement for now is to have a dataset downloaded in below location
        downloaded_files = {"train": "/Users/rafalpowalski/datasets/docvqa/train/train_v1.0.json",
                            "val": "/Users/rafalpowalski/datasets/docvqa/val/val_v1.0.json"}

        return [
            # datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["val"]}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        ocr_path_dir = Path(filepath).parent / "ocr_results"
        image_dir = Path(filepath).parent / "documents"
        with open(filepath, encoding="utf-8") as f:
            ds = json.load(f)

        # TODO: maybe it should be per document_id which will have multiple question answer pairs?

        for example in ds["data"]:

            image_id = example["image"].split('/')[-1].split('.')[0]
            ocr_path = ocr_path_dir / f"{image_id}.json"
            # TODO: add image to the example
            # image_path = image_dir / f"{image_id}.png"
            # img = cv2.imread(str(image_path))
            doc = load_msocr_file(ocr_path)
            qid = str(example["questionId"])

            # Features currently used are "context", "question", and "answers".
            # Others are extracted here for the ease of future expansions.
            yield qid, {
                "id": qid,
                "image_id": image_id,
                "words": doc.words,
                "bboxes": doc.bboxes,
                "page_bboxes": doc.page_bboxes,
                "page_spans": doc.page_spans,
                "question": example["question"],
                "answer": example["answers"][0],
            }
