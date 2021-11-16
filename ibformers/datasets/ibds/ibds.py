import json
import logging
import os
import urllib
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union, NamedTuple, Optional, Callable, Sequence
import datasets
from datasets import BuilderConfig, Features, config
from datasets.fingerprint import Hasher

from instabase.ocr.client.libs.ibocr import (
    ParsedIBOCRBuilder,
    IBOCRRecord,
    IBOCRRecordLayout,
    ParsedIBOCR,
)
from instabase.ocr.client.libs.ocr_types import WordPolyDict
from typing_extensions import TypedDict, Literal
from more_itertools import consecutive_groups
import numpy as np
from ibformers.data.utils import ImageProcessor

_DESCRIPTION = """\
Internal Instabase Dataset format organized into set of IbDoc files and IbAnnotator file.
"""


def get_open_fn(ibsdk):
    return open if ibsdk is None else ibsdk.ibopen


# Define some helper classes for easier typing
BoundingBox = Tuple[float, float, float, float]
Span = Tuple[int, int]
AnnotationLabelId = str


class AnnotationLayout(TypedDict):
    width: float
    height: float


class AnnotationRect(TypedDict):
    x: int
    y: int
    w: int
    h: int


class AnnotationLabel(TypedDict):
    id: str
    name: str
    type: str


class AnnotationPosition(TypedDict):
    rect: AnnotationRect
    page: int


class AnnotationWordMetadata(TypedDict):
    dataType: Literal["WordPoly"]
    type: Literal["rect"]
    rawWord: str
    position: AnnotationPosition


class Annotation(TypedDict):
    value: str
    metadata: List[Union[AnnotationWordMetadata, Any]]


class AnnotationFile(TypedDict):
    id: str
    inputPath: str
    ocrPath: str
    isInputOcr: bool
    pages: Dict[str, AnnotationLayout]
    annotations: Dict[AnnotationLabelId, Annotation]


class Annotation(TypedDict):
    value: str
    metadata: List[Union[AnnotationWordMetadata, Any]]


class _SearchKey(NamedTuple):
    x: float
    y: float
    page: int
    word: str


class LabelWithId(TypedDict):
    id: int
    text: str


class LabelEntity(TypedDict):
    name: str
    order_id: int
    text: str
    char_spans: List
    token_spans: List
    token_label_id: int


class IbDsBuilderConfig(BuilderConfig):
    """Base class for :class:`DatasetBuilder` data configuration.
    Copied from BuilderConfig class
    Contains modifications which crete custom config_id (fingerprint) based on the ibannotator content
    """

    def __init__(self, *args, **kwargs):
        super(IbDsBuilderConfig, self).__init__(*args, **kwargs)

    def create_config_id(
        self,
        config_kwargs: dict,
        custom_features: Optional[Features] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> str:
        """
        The config id is used to build the cache directory.
        By default it is equal to the config name.
        However the name of a config is not sufficient to have a unique identifier for the dataset being generated
        since it doesn't take into account:
        - the config kwargs that can be used to overwrite attributes
        - the custom features used to write the dataset
        - the data_files for json/text/csv/pandas datasets
        Therefore the config id is just the config name with an optional suffix based on these.
        """
        # Possibly add a suffix to the name to handle custom features/data_files/config_kwargs
        suffix: Optional[str] = None
        config_kwargs_to_add_to_suffix = config_kwargs.copy()
        # name and version are already used to build the cache directory
        config_kwargs_to_add_to_suffix.pop("name", None)
        config_kwargs_to_add_to_suffix.pop("version", None)
        config_kwargs_to_add_to_suffix.pop("ibsdk", None)
        # data files are handled differently
        config_kwargs_to_add_to_suffix.pop("data_files", None)
        if (
            "data_dir" in config_kwargs_to_add_to_suffix
            and config_kwargs_to_add_to_suffix["data_dir"] is None
        ):
            del config_kwargs_to_add_to_suffix["data_dir"]
        if config_kwargs_to_add_to_suffix:
            # we don't care about the order of the kwargs
            config_kwargs_to_add_to_suffix = {
                k: config_kwargs_to_add_to_suffix[k] for k in sorted(config_kwargs_to_add_to_suffix)
            }
            if all(
                isinstance(v, (str, bool, int, float))
                for v in config_kwargs_to_add_to_suffix.values()
            ):
                suffix = ",".join(
                    str(k) + "=" + urllib.parse.quote_plus(str(v))
                    for k, v in config_kwargs_to_add_to_suffix.items()
                )
                if len(suffix) > 32:  # hash if too long
                    suffix = Hasher.hash(config_kwargs_to_add_to_suffix)
            else:
                suffix = Hasher.hash(config_kwargs_to_add_to_suffix)

        if "train" in self.data_files:
            m = Hasher()
            if suffix:
                m.update(suffix)

            # stat_fn = self.ibsdk.stat if self.ibsdk is not None else os.path.getmtime
            # stat = str(stat_fn(self.data_files["train"]))
            open_fn = get_open_fn(self.ibsdk)
            with open_fn(self.data_files["train"], "r") as annotation_file:
                content = json.load(annotation_file)
                fingerprint_content = {
                    k: v for k, v in content.items() if k in ("files", "labels", "testFiles")
                }
            m.update(self.data_files["train"])
            m.update(fingerprint_content)
            suffix = m.hexdigest()

        if custom_features is not None:
            m = Hasher()
            if suffix:
                m.update(suffix)
            m.update(custom_features)
            suffix = m.hexdigest()

        if suffix:
            config_id = self.name + "-" + suffix
            if len(config_id) > config.MAX_DATASET_CONFIG_ID_READABLE_LENGTH:
                config_id = self.name + "-" + Hasher.hash(suffix)
            return config_id
        else:
            return self.name


def _read_parsedibocr(builder: ParsedIBOCR) -> Tuple[List[WordPolyDict], List[IBOCRRecordLayout]]:
    """Open an ibdoc or ibocr using the ibfile and return the words and layout information for each page"""
    words = []
    layouts = []
    record: IBOCRRecord
    # Assuming each record is a page in order and each record is single-page
    # Assuming nothing weird is going on with page numbers
    for record in builder.get_ibocr_records():
        words += [i for j in record.get_lines() for i in j]
        l = record.get_metadata_list()
        layouts.extend([i.get_layout() for i in l])

    assert all(
        word["page"] in range(len(layouts)) for word in words
    ), "Something with the page numbers went wrong"

    return words, layouts


def process_labels_from_annotation(
    words: List[WordPolyDict],
    annotation_file: Optional[AnnotationFile] = None,
    label2id: Optional[Dict[str, int]] = None,
    ann_label_id2label: Optional[Dict[AnnotationLabelId, str]] = None,
) -> Tuple[List[LabelEntity], Sequence[int]]:

    token_label_ids = np.zeros((len(words)), dtype=np.int64)
    entities = []
    if annotation_file is not None:
        label2ann_label_id = {v: k for k, v in ann_label_id2label.items()}

    key_to_words = {
        _SearchKey(x["start_x"], x["start_y"], x["page"], x["raw_word"]): i
        for i, x in enumerate(words)
    }
    if len(key_to_words) != len(words):
        logging.error(
            f"Issue with assumption that _SearchKey(x, y, page, word) is unique. "
            f"{len(key_to_words)}!={len(words)}"
        )
    annotation: Annotation
    for label_name, lab_id in label2id.items():
        if lab_id == 0 and label_name == "O":
            continue
        if annotation_file is None:
            # add empty entities if there is no annotations
            entity: LabelEntity = LabelEntity(
                name=label_name,
                order_id=0,
                text="",
                char_spans=[],
                token_spans=[],
                token_label_id=lab_id,
            )

        else:
            ann_label_id = label2ann_label_id[label_name]
            annotation = annotation_file["annotations"].get(
                ann_label_id, {"value": "", "metadata": []}
            )

            metadata = annotation["metadata"]
            word_metadata: AnnotationWordMetadata
            label_words = []
            for word_metadata in metadata:
                if word_metadata["dataType"] != "WordPoly":
                    raise ValueError("Unexpected (non-wordpoly) annotation. Skipping.")

                pos = word_metadata["position"]
                rect: AnnotationRect = pos["rect"]
                key = _SearchKey(rect["x"], rect["y"], pos["page"], word_metadata["rawWord"])
                if key not in key_to_words:
                    raise RuntimeError(
                        f"Couldn't find word {repr(key)} in document {annotation_file['ocrPath']}."
                    )
                word_id_global = key_to_words[key]

                word_text = word_metadata["rawWord"]
                assert (
                    word_text.strip() == words[word_id_global]["word"].strip()
                ), "Annotation does not match with document words"

                label_words.append(LabelWithId(id=word_id_global, text=word_text))

            # TODO: information about multi-item entity separation should be obtained during annotation
            # group consecutive words as the same entity occurrence
            label_words.sort(key=lambda x: x["id"])
            label_groups = [
                list(group) for group in consecutive_groups(label_words, ordering=lambda x: x["id"])
            ]
            # create spans for groups, span will be created by getting id for first and last word in the group
            label_token_spans = [[group[0]["id"], group[-1]["id"] + 1] for group in label_groups]

            for span in label_token_spans:
                token_label_ids[span[0] : span[1]] = lab_id

            entity: LabelEntity = LabelEntity(
                name=label_name,
                order_id=0,
                text=annotation["value"],
                char_spans=[],
                token_spans=label_token_spans,
                token_label_id=lab_id,
            )
        entities.append(entity)

    return entities, token_label_ids


def get_images_from_layouts(
    layouts: List[IBOCRRecordLayout],
    image_processor: ImageProcessor,
    ocr_path: str,
    open_fn: Callable,
    page_nums: List[int],
):
    """
    Optionally get images and save it as array objects
    :param layouts: list of layouts (pages) objects
    :param image_processor: callable object used to process images
    :param ocr_path: path of ocr used to obtain relative path of images
    :param open_fn: fn used to open the image
    :param page_nums: number of pages associated with given record
    :return: image array
    """
    img_lst = []

    for page in page_nums:
        lay = layouts[page]
        img_path = Path(lay.get_processed_image_path())
        try:
            with open_fn(str(img_path)) as img_file:
                img_arr = image_processor(img_file).astype(np.uint8)
        except OSError:
            # try relative path - useful for debugging
            ocr_path = Path(ocr_path)
            img_rel_path = ocr_path.parent.parent / "s1_process_files" / "images" / img_path.name
            with open_fn(str(img_rel_path), "rb") as img_file:
                img_arr = image_processor(img_file).astype(np.uint8)

        img_lst.append(img_arr)

    img_arr_all = np.stack(img_lst, axis=0)

    return img_arr_all


def process_parsedibocr(
    parsedibocr: ParsedIBOCR,
    open_fn,
    use_image: bool,
    image_processor: ImageProcessor,
    doc_annotations: Optional[AnnotationFile] = None,
    label2id: Optional[Dict[str, int]] = None,
    ann_label_id2label: Optional[Dict[AnnotationLabelId, str]] = None,
):
    """
    prepare examples based on annotations and ocr information
    :param image_processor:
    :param doc_annotations: content of ibannotator file
    :param label2id: dictionary which contain mapping from the entity name to class_id
    :param ann_label_id2label: mapping from entity id used in annotation file to entity name
    :param use_image: whether to load an image as a feature
    :return: Dictionary containing words, bboxes and per-word annotations
    """

    words, layouts = _read_parsedibocr(parsedibocr)
    doc_id = (
        parsedibocr.get_document_path(0)[0]
        if doc_annotations is None
        else doc_annotations["ocrPath"]
    )

    assert (
        doc_id is not None and doc_id != ""
    ), "An issue occured while obtaining a document path from an ibocr"
    record: IBOCRRecord

    # get content of the WordPolys
    word_lst: List[str] = [w["word"] for w in words]
    bbox_arr = np.array([[w["start_x"], w["start_y"], w["end_x"], w["end_y"]] for w in words])
    word_pages_arr = np.array([w["page"] for w in words])

    # get number of tokens for each page,
    # below is a bit overcomplicated because there might be empty pages
    page_nums, page_token_counts = np.unique(word_pages_arr, return_counts=True)
    page_ct_arr = np.zeros(len(layouts), dtype=np.int64)  # assume that layouts are single pages
    for page_num, page_token_ct in zip(page_nums, page_token_counts):
        page_ct_arr[page_num] = page_token_ct

    # get page spans
    page_offsets = np.cumsum(page_ct_arr)
    page_spans = np.stack((np.pad(page_offsets[:-1], (1, 0)), page_offsets), axis=-1)

    # get page height and width
    page_bboxes = np.array([[0, 0, pg.get_width(), pg.get_height()] for pg in layouts])

    # normalize bboxes - divide only by width to keep information about ratio
    width_per_token = np.take(page_bboxes[:, 2], word_pages_arr)
    norm_bboxes = bbox_arr * 1000 / width_per_token[:, None]
    norm_page_bboxes = page_bboxes * 1000 / page_bboxes[:, 2:3]

    entities, token_label_ids = process_labels_from_annotation(
        annotation_file=doc_annotations,
        words=words,
        label2id=label2id,
        ann_label_id2label=ann_label_id2label,
    )

    features = {
        "id": doc_id,
        "words": word_lst,
        "bboxes": norm_bboxes,
        "word_original_bboxes": bbox_arr,
        "word_page_nums": word_pages_arr,
        "page_bboxes": norm_page_bboxes,
        "page_spans": page_spans,
        "token_label_ids": token_label_ids,
        "entities": entities,
    }

    if use_image:
        page_nums = np.unique(word_pages_arr)
        page_nums.sort()
        images = get_images_from_layouts(layouts, image_processor, doc_id, open_fn, page_nums)
        # assert len(norm_page_bboxes) == len(images), "Number of images should match number of pages in document"
        features["images"] = images
        features["images_page_nums"] = page_nums

    return features


class IbDsConfig(IbDsBuilderConfig):
    """
    Config for Instabase Format Datasets
    """

    def __init__(self, use_image=False, ibsdk=None, id2label=None, **kwargs):
        """BuilderConfig for Instabase Format Datasets.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(IbDsConfig, self).__init__(**kwargs)
        self.use_image = use_image
        self.ibsdk = ibsdk
        self.id2label = id2label


class IbDs(datasets.GeneratorBasedBuilder):
    """
    Instabase internal dataset format, creation of dataset can be done by passing ibannotator file location or
    in the inference mode passing the list of parsedibocr files
    """

    BUILDER_CONFIGS = [
        IbDsConfig(
            name="ibds", version=datasets.Version("1.0.0"), description="Instabase Format Datasets"
        ),
    ]

    def __init__(self, *args, **kwargs):
        super(IbDs, self).__init__(*args, **kwargs)
        self.image_processor = (
            ImageProcessor(do_resize=True, size=224) if self.config.use_image else None
        )
        self.ann_label_id2label = None

    def _info(self):

        # get schema of the the dataset
        # TODO(ibds): Check if schema can be saved in the separate file, so we don't load whole annotation file
        data_files = self.config.data_files
        assert len(data_files) == 1, "Only one annotation path should be provided"
        assert isinstance(data_files, dict), "data_files argument should be a dict for this dataset"
        if "train" in data_files:
            annotation_path = data_files["train"]
            open_fn = get_open_fn(self.config.ibsdk)
            with open_fn(annotation_path, "r") as annotation_file:
                labels = json.load(annotation_file)["labels"]
            classes = ["O"] + [lab["name"] for lab in labels]
        elif "test" in data_files:
            raise NotImplementedError(
                "Inference mode for ibds is not longer supported. Use docpro_ds."
            )
        else:
            raise ValueError("data_file argument should be either in train or test mode")

        ds_features = {
            "id": datasets.Value("string"),
            "words": datasets.Sequence(datasets.Value("string")),
            "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=4)),
            # needed to generate prediction file, after evaluation
            "word_original_bboxes": datasets.Sequence(
                datasets.Sequence(datasets.Value("float32"), length=4)
            ),
            "word_page_nums": datasets.Sequence(datasets.Value("int32")),
            "page_bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=4)),
            "page_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
            "token_label_ids": datasets.Sequence(datasets.features.ClassLabel(names=classes)),
            # Do not output entities as this ds is used only by SL models by now
            "entities": datasets.Sequence(
                {
                    "name": datasets.Value("string"),  # change to id?
                    "order_id": datasets.Value(
                        "int64"
                    ),  # not supported yet, annotation app need to implement it
                    "text": datasets.Value("string"),
                    "char_spans": datasets.Sequence(
                        datasets.Sequence(datasets.Value("int32"), length=2)
                    ),
                    "token_spans": datasets.Sequence(
                        datasets.Sequence(datasets.Value("int32"), length=2)
                    ),
                    "token_label_id": datasets.Value("int64"),
                }
            ),
        }

        if self.config.use_image:
            # first dimension is defined as a number of pages in the document
            # workaround with sequences - dynamic dimensions are not yet supported by hf/datasets
            # ds_features['images'] = datasets.Array4D(shape=(None, 3, 224, 224), dtype="uint8")
            # ds_features['images'] = datasets.Sequence(datasets.Sequence(
            #     datasets.Sequence(datasets.Sequence(datasets.Value('uint8'), length=224), length=224), length=3))
            # for now support only 1-page documents
            ds_features["images"] = datasets.Array4D(shape=(None, 3, 224, 224), dtype="uint8")
            ds_features["images_page_nums"] = datasets.Sequence(datasets.Value("int32"))

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=datasets.Features(ds_features),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        data_files = self.config.data_files
        if "train" in data_files:
            annotation_path = data_files["train"]
            open_fn = get_open_fn(self.config.ibsdk)
            with open_fn(annotation_path, "r") as annotation_file:
                annotations = json.load(annotation_file)

            self.ann_label_id2label = {lab["id"]: lab["name"] for lab in annotations["labels"]}

            # create generators for train and test
            train_files = (
                file
                for file in annotations["files"]
                if file["id"] not in annotations["testFiles"] and len(file["annotations"]) > 0
            )
            val_files = (
                file
                for file in annotations["files"]
                if file["id"] in annotations["testFiles"] and len(file["annotations"]) > 0
            )
            # test set is the sum of unannotated documents and validation set
            test_files = (
                file
                for file in annotations["files"]
                if len(file["annotations"]) == 0 or file["id"] in annotations["testFiles"]
            )

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN, gen_kwargs={"files": train_files, "open_fn": open_fn}
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"files": val_files, "open_fn": open_fn},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST, gen_kwargs={"files": test_files, "open_fn": open_fn}
                ),
            ]

        elif "test" in data_files:
            # inference input is a list of parsedibocr files
            raise NotImplementedError(
                "Inference mode for ibds is not longer supported. Use docpro_ds."
            )
        else:
            raise ValueError("data_file argument should be either in train or test mode")

    def _generate_examples(self, files, open_fn=None):
        """Yields examples."""
        label2id = self.info.features["token_label_ids"].feature._str2int

        for file in files:
            if isinstance(file, dict):
                # open file based on the Path
                ocr_path = file["ocrPath"]
                if not (ocr_path.endswith(".ibdoc") or ocr_path.endswith(".ibocr")):
                    raise ValueError(f"Invaild document path: {ocr_path}")

                try:
                    with open_fn(ocr_path, "rb") as f:
                        data = f.read()
                except FileNotFoundError:
                    # change to relative path to annotation path
                    logging.warning(
                        f'Didnt find absolute path from ibannotator. Trying relative path'
                    )
                    annotator_path = Path(self.config.data_files['train'])
                    fallback_path = annotator_path.parent / Path(*Path(ocr_path).parts[-4:])
                    if not fallback_path.is_file():
                        logging.error(
                            f"Both absolute path {ocr_path} and relative path {fallback_path} not found"
                        )
                    with open_fn(fallback_path, "rb") as f:
                        data = f.read()

                builder: ParsedIBOCRBuilder
                builder, err = ParsedIBOCRBuilder.load_from_str(ocr_path, data)

                if err:
                    builder, err = ParsedIBOCRBuilder.load_from_str(ocr_path, data)  # for debugging
                    logging.warning("Could not load file: {}".format(ocr_path))
                    continue
                    # IOError("Could not load file: {}".format(ocr_path))
                ibocr = builder.as_parsed_ibocr()
                annotations = file

            elif isinstance(file, ParsedIBOCR):
                ibocr = file
                annotations = None
            else:
                raise RuntimeError("Encounter not supported file format")

            doc_dict = process_parsedibocr(
                ibocr,
                open_fn,
                self.config.use_image,
                self.image_processor,
                annotations,
                label2id,
                self.ann_label_id2label,
            )

            yield doc_dict["id"], doc_dict
