import logging
import time
import traceback
import urllib
from abc import abstractmethod, ABCMeta
from io import BytesIO
from pathlib import Path
from typing import Tuple, List, Callable, Optional, Union, Any, Dict, Iterable, Type

import datasets
import numpy as np
from datasets import BuilderConfig, Features
from datasets.fingerprint import Hasher
from datasets.config import MAX_DATASET_CONFIG_ID_READABLE_LENGTH

from ibformers.data.utils import ImageProcessor
from instabase.ocr.client.libs.ibocr import IBOCRRecordLayout, IBOCRRecord


BoundingBox = Tuple[float, float, float, float]
Span = Tuple[int, int]
IBDS_WRITER_BATCH_SIZE = 8


_DESCRIPTION = """\
Internal Instabase Dataset format organized into set of IbDoc files.
"""


class IBDSBuilderConfig(BuilderConfig):
    """Base class for :class:`DatasetBuilder` data configuration.
    Copied from BuilderConfig class
    Contains modifications which create custom config_id (fingerprint) based on the dataset.json content
    """

    def __init__(self, *args, **kwargs):
        super(IBDSBuilderConfig, self).__init__(*args, **kwargs)

    def create_config_id(
        self,
        config_kwargs: dict,
        custom_features: Optional[Features] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> str:

        # Possibly add a suffix to the name to handle custom features/data_files/config_kwargs
        suffix: Optional[str] = None
        config_kwargs_to_add_to_suffix = config_kwargs.copy()
        # name and version are already used to build the cache directory
        config_kwargs_to_add_to_suffix.pop("name", None)
        config_kwargs_to_add_to_suffix.pop("version", None)
        config_kwargs_to_add_to_suffix.pop("ibsdk", None)
        # data files are handled differently
        config_kwargs_to_add_to_suffix.pop("data_files", None)
        if "data_dir" in config_kwargs_to_add_to_suffix and config_kwargs_to_add_to_suffix["data_dir"] is None:
            del config_kwargs_to_add_to_suffix["data_dir"]
        if config_kwargs_to_add_to_suffix:
            # we don't care about the order of the kwargs
            config_kwargs_to_add_to_suffix = {
                k: config_kwargs_to_add_to_suffix[k] for k in sorted(config_kwargs_to_add_to_suffix)
            }
            if all(isinstance(v, (str, bool, int, float)) for v in config_kwargs_to_add_to_suffix.values()):
                suffix = ",".join(
                    str(k) + "=" + urllib.parse.quote_plus(str(v)) for k, v in config_kwargs_to_add_to_suffix.items()
                )
                if len(suffix) > 32:  # hash if too long
                    suffix = Hasher.hash(config_kwargs_to_add_to_suffix)
            else:
                suffix = Hasher.hash(config_kwargs_to_add_to_suffix)

        if "train" in self.data_files:
            m = Hasher()
            if suffix:
                m.update(suffix)

            dataset_list = load_datasets(self.data_files["train"], self.ibsdk)
            # build fingerprint based on chosen metadata items, changing any of below fields would cause dataset to be rebuild
            fingerprint_content = sorted(
                [
                    str(v)
                    for ds in dataset_list
                    for k, v in ds.metadata.items()
                    if k in ("id", "last_edited", "last_editor")
                ]
            )
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
            if len(config_id) > MAX_DATASET_CONFIG_ID_READABLE_LENGTH:
                config_id = self.name + "-" + Hasher.hash(suffix)
            return config_id
        else:
            return self.name


class ExtractionMode:
    TEXT = "text"
    TABLE = "table"


class IBDSConfig(IBDSBuilderConfig):
    """
    Config for Instabase Datasets which contain additional attributes related to Doc Pro dataset
    :param use_image: whether to load the image of the page
    :param ibsdk: if images is used ibsdk is used to pick up the image
    :param id2label: label id to label name mapping
    :param extraction_class_name: name of the extracted class, for classification it will be empty
    :param kwargs: keyword arguments forwarded to super.
    """

    def __init__(
        self,
        use_image=False,
        ibsdk=None,
        id2label=None,
        table_id2label=None,
        extraction_class_name=None,
        norm_bboxes_to_max: bool = False,
        bbox_scale_factor: int = 1000,
        extraction_mode: str = ExtractionMode.TEXT,
        **kwargs,
    ):
        super(IBDSConfig, self).__init__(**kwargs)
        self.use_image = use_image
        self.ibsdk = ibsdk
        self.id2label = id2label
        self.table_id2label = table_id2label
        self.extraction_class_name = extraction_class_name
        self.norm_bboxes_to_max = norm_bboxes_to_max
        self.bbox_scale_factor = bbox_scale_factor
        self.extraction_mode = extraction_mode

    @property
    def label2id(self) -> Optional[Dict[int, str]]:
        return {v: k for k, v in self.id2label.items()} if self.id2label is not None else None

    @property
    def table_label2id(self) -> Optional[Dict[int, str]]:
        return {v: k for k, v in self.table_id2label.items()} if self.table_id2label is not None else None

    @property
    def classes(self) -> Optional[List[str]]:
        return [self.id2label[i] for i in range(len(self.id2label))] if self.id2label is not None else None


class IbDs(datasets.GeneratorBasedBuilder, metaclass=ABCMeta):
    """
    Instabase internal dataset format, creation of dataset can be done by passing list of datasets
    """

    CONFIG_NAME = "ib_ds2"
    BUILDER_CONFIG_CLASS = IBDSConfig
    DEFAULT_WRITER_BATCH_SIZE = IBDS_WRITER_BATCH_SIZE

    BUILDER_CONFIGS = [
        IBDSConfig(
            name=CONFIG_NAME,
            version=datasets.Version("1.0.0"),
            description="Instabase Format Datasets",
        ),
    ]

    def __init__(self, *args, **kwargs):
        super(IbDs, self).__init__(*args, **kwargs)
        self.image_processor = self.create_image_processor(self.config)
        self.extraction_class_name = self.config.extraction_class_name

    def _info(self):

        data_files = self.config.data_files
        assert isinstance(data_files, dict), "data_files argument should be a dict for this dataset"
        if "train" in data_files:
            features = self.get_train_dataset_features()
        elif "test" in data_files:
            features = self.get_inference_dataset_features(self.config)
        else:
            raise ValueError("data_file argument should be either in train or test mode")

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
        )

    @abstractmethod
    def get_train_dataset_features(self) -> datasets.Features:
        pass

    @classmethod
    @abstractmethod
    def get_inference_dataset_features(cls, config: IBDSConfig) -> datasets.Features:
        pass

    @classmethod
    def create_image_processor(cls, config: IBDSConfig) -> Optional[ImageProcessor]:
        return ImageProcessor(do_resize=True, size=224) if config.use_image else None

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        data_files = self.config.data_files
        if "train" in data_files:
            datasets_list = load_datasets(data_files["train"], self.config.ibsdk)
            annotation_items = self._get_annotation_generator(datasets_list)
            # TODO: implement functionality in Dataset SDK to get iterators for train and test files separately

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"annotation_items": annotation_items},
                )
            ]

        elif "test" in data_files:
            # inference input is a list of parsedibocr files
            test_files = data_files["test"]
            annotation_items = self._get_annotation_from_model_service(test_files, self.config)

            return [
                datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"annotation_items": annotation_items}),
            ]

    @classmethod
    @abstractmethod
    def process_item(cls, item, config, image_processor) -> Dict:
        pass

    @abstractmethod
    def _get_annotation_generator(self, datasets_list: List["DatasetSDK"]) -> Iterable[Any]:
        pass

    @classmethod
    @abstractmethod
    def _get_annotation_from_model_service(cls, records: List[Any], config) -> Iterable[Any]:
        pass

    @classmethod
    def get_examples_from_model_service(cls, records: List[Any], config, image_processor):
        examples = []
        for item in cls._get_annotation_from_model_service(records, config):
            doc_dict = cls.process_item(item, config, image_processor)
            if doc_dict is not None:
                examples.append(doc_dict)
        return examples

    def _generate_examples(self, annotation_items):
        """Yields examples."""
        # these items are yielded by _get_annotation_generator
        for item in annotation_items:
            doc_dict = self.process_item(item, self.config, self.image_processor)
            if doc_dict is not None:
                yield doc_dict["id"], doc_dict


def local_read_fn(file: Union[str, Path]) -> bytes:
    return Path(file).read_bytes()


def get_open_fn(ibsdk) -> Callable[[Union[str, Path]], bytes]:
    return local_read_fn if ibsdk is None else ibsdk.read_file


def get_common_feature_schema(use_image):
    """
    Get common schema for ocr features for Ib datasets
    :param use_image: whether to add features related to image
    :return: Dict with schema
    """
    ds_features = {
        "id": datasets.Value("string"),
        "split": datasets.Value("string"),
        # to distiguish test files marked in the doc pro app from all files for which predictions are needed
        "is_test_file": datasets.Value("bool"),
        # TODO: remove this column once Dataset SDK will allow for test files iterators
        "words": datasets.Sequence(datasets.Value("string")),
        "bboxes": datasets.Array2D(shape=(None, 4), dtype="int32"),
        # needed to generate prediction file, after evaluation
        "word_original_bboxes": datasets.Array2D(shape=(None, 4), dtype="float32"),
        "word_page_nums": datasets.Sequence(datasets.Value("int32")),
        "word_line_idx": datasets.Sequence(datasets.Value("int32")),
        "word_in_line_idx": datasets.Sequence(datasets.Value("int32")),
        "page_bboxes": datasets.Array2D(shape=(None, 4), dtype="int32"),
        "page_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
    }
    if use_image:
        # first dimension is defined as a number of pages in the document
        ds_features["images"] = datasets.Array4D(shape=(None, 3, 224, 224), dtype="uint8")
        ds_features["images_page_nums"] = datasets.Sequence(datasets.Value("int32"))

    return ds_features


def load_datasets(dataset_paths: List[str], ibsdk: Any) -> List["DatasetSDK"]:
    """
    Load datasets base on the list of paths. Can be either Remote or Local
    :param dataset_paths:
    :param ibsdk: sdk used to open remote files
    :return: List of DatasetSDK
    """
    # use lazy imports as in model service sdk is not available
    try:
        from instabase.dataset_utils.sdk import RemoteDatasetSDK, LocalDatasetSDK
    except ImportError as err:
        logging.error(f"SDK not found: {err}")
    assert isinstance(dataset_paths, list)

    if ibsdk is None:
        file_client = None
        username = None
    else:
        file_client = ibsdk.file_client
        username = ibsdk.username
    try:
        # load from doc pro
        if file_client is None:
            datasets_list = [LocalDatasetSDK(dataset_path) for dataset_path in dataset_paths]
        else:
            datasets_list = [RemoteDatasetSDK(dataset_path, file_client, username) for dataset_path in dataset_paths]

    except Exception as e:
        logging.error(traceback.format_exc())
        raise RuntimeError(f"Error while compiling the datasets: {e}") from e

    return datasets_list


def get_image_features(
    ocr_features,
    layouts: List[IBOCRRecordLayout],
    ocr_path: str,
    open_fn: Any,
    image_processor: ImageProcessor,
):
    """
    Optionally get images and save it as array objects
    :param ocr_features: Dict of features obtained from ocr
    :param layouts: list of layouts (pages) objects
    :param ocr_path: path of ocr used to obtain relative path of images
    :param open_fn: fn used to open the files. Can be either local files or files on Instabase FS
    :param image_processor: ImageProcessor used to process the image to array
    :return: Dict containing image array and page numbers
    """
    page_nums = np.unique(ocr_features["word_page_nums"])
    page_nums.sort()

    img_lst = []

    for page in page_nums:
        lay = layouts[page]
        img_path = Path(lay.get_processed_image_path())
        # try relative path - useful if dataset was moved
        ocr_path = Path(ocr_path)
        img_rel_path = ocr_path.parent.parent / "s1_process_files" / "images" / img_path.name
        img_arr = get_image(img_path, img_rel_path, open_fn, image_processor)

        if img_arr is None:
            raise OSError(
                f"Image does not exist in the image_path location: {img_path}. "
                f"It was also not found in the location relative to ibdoc: {img_rel_path}. "
                f"Script also waited for images to be saved for 30s"
            )

        img_lst.append(img_arr)

    img_arr_all = np.stack(img_lst, axis=0)

    return {"images": img_arr_all, "images_page_nums": page_nums}


def get_image(img_path: str, img_rel_path: str, open_fn: Any, image_processor: ImageProcessor) -> np.ndarray:
    """
    Get image data based on the paths
    :param img_path: absolute path of the image page
    :param img_rel_path: path relative to actual path of ibdoc
    :param open_fn: fn used to open the files. Can be either local files or files on Instabase FS
    :param image_processor: ImageProcessor used to process the image to array
    :return: np.ndarray with the image
    """
    # wait for 30s in case of image is saved async in the flow
    # TODO: remove it once we integrate model-service with datastore - this is temporary workaround
    time_waited = 0
    img_arr = None
    for i in range(16):
        try:
            img_bytes = open_fn(str(img_path))
            img_file = BytesIO(img_bytes)
            img_arr = image_processor(img_file).astype(np.uint8)
            break
        except OSError:
            try:
                img_bytes = open_fn(str(img_rel_path))
                img_file = BytesIO(img_bytes)
                img_arr = image_processor(img_file).astype(np.uint8)
                break
            except OSError:
                time.sleep(2)
                time_waited += 2

    if time_waited > 0 and img_arr is not None:
        logging.warning(f"Script waited for {time_waited} sec for image {img_path} to be saved")

    return img_arr


def prepare_word_pollys_and_layouts_for_record(record: IBOCRRecord) -> Tuple[List, List]:
    """
    Extract word pollys and layouts from the record
    :param record: IBOCRRecord
    :return: List of word pollys and List of layouts
    """
    lines = record.get_lines()
    layouts = [mtd.get_layout() for mtd in record.get_metadata_list()]
    words = []
    for line_idx, line in enumerate(lines):
        for word_idx, word in enumerate(line):
            word["line_index"] = line_idx
            word["word_index"] = word_idx
            words.append(word)

    return words, layouts


def get_ocr_features(
    words: List, layouts: List, doc_id: str, norm_bboxes_to_max: bool = False, bbox_scale_factor: int = 1000
) -> Dict:
    """

    :param words: List of WordPolyDict
    :param layouts: List of IBOCRRecordLayout
    :param doc_id: Identifier of the record
    :return: Dict of features
    """
    # get content of the WordPolys
    word_lst: List[str] = [w["word"] for w in words]
    bbox_arr = np.array([[w["start_x"], w["start_y"], w["end_x"], w["end_y"]] for w in words])
    word_pages_arr = np.array([w["page"] for w in words], dtype=np.int32)
    word_line_idx_arr = np.array([w["line_index"] for w in words], dtype=np.int32)
    word_in_line_idx_arr = np.array([w["word_index"] for w in words], dtype=np.int32)

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
    # normalize bboxes -
    size_per_token = np.take(page_bboxes[:, 2:], word_pages_arr, axis=0)
    if norm_bboxes_to_max:
        word_bbox_norm_factor = size_per_token.max(1)[:, None]
        page_bbox_norm_factor = page_bboxes[:, 2:].max(1)[:, None]
    else:
        # divide only by width to keep information about ratio
        word_bbox_norm_factor = size_per_token[:, 0][:, None]
        page_bbox_norm_factor = page_bboxes[:, 2][:, None]

    # Validate bboxes
    fix_bbox_arr = validate_and_fix_bboxes(bbox_arr, size_per_token, word_pages_arr, page_bboxes, doc_id)
    norm_bboxes = fix_bbox_arr * bbox_scale_factor / word_bbox_norm_factor
    norm_page_bboxes = page_bboxes * bbox_scale_factor / page_bbox_norm_factor

    features = {
        "words": word_lst,
        "bboxes": norm_bboxes,
        "word_original_bboxes": bbox_arr,
        "word_page_nums": word_pages_arr,
        "word_line_idx": word_line_idx_arr,
        "word_in_line_idx": word_in_line_idx_arr,
        "page_bboxes": norm_page_bboxes,
        "page_spans": page_spans,
    }

    return features


def validate_and_fix_bboxes(bbox_arr, page_size_per_token, word_pages_arr, page_bboxes, doc_id):
    """
    Checks whether bboxes are correct. If not, tries to correct it
    """
    if len(bbox_arr) == 0:
        return bbox_arr
    fixed_arr = bbox_arr
    for dim in range(2):
        tokens_outside_dim = np.nonzero(fixed_arr[:, 2 + dim] > page_size_per_token[:, dim])
        if len(tokens_outside_dim[0]) > 0:
            example_idx = tokens_outside_dim[0][0]
            ex_bbox = bbox_arr[example_idx]
            ex_page = page_bboxes[word_pages_arr[example_idx]]
            logging.error(
                f"found bboxes outside of the page for {doc_id}. Example bbox {ex_bbox} page:({ex_page})."
                f"These will be trimmed to page coordinates. Please review your OCR settings."
            )
            # fixing bboxes
            # use tile to double last dim size and apply trimming both to x1,y1 and x2,y2
            fixed_arr = np.minimum(fixed_arr, np.tile(page_size_per_token, 2))

        tokens_negative = np.nonzero(fixed_arr < 0)
        if len(tokens_negative[0]) > 0:
            example_idx = tokens_negative[0][0]
            ex_bbox = bbox_arr[example_idx]
            ex_page = page_bboxes[word_pages_arr[example_idx]]
            logging.error(
                f"found bboxes with negative coord of the page for {doc_id}. Example bbox {ex_bbox} page:({ex_page})."
                f"These will be trimmed to page coordinates. Please review your OCR settings."
            )

            fixed_arr = np.maximum(fixed_arr, 0)
    return fixed_arr


def assert_valid_record(ibocr_record: IBOCRRecord) -> Optional[str]:
    """
    Confirms an IBOCR Record's lines and text are aligned, such that
    provenance tracking will work downstream.
    Returns a string describing an error if the record is malformed, or
    None if the record is well-formed.
    """
    text = ibocr_record.get_text()
    split_text = text.split("\n")

    lines = ibocr_record.get_lines()

    len_split_text = len(split_text)
    len_lines = len(lines)
    if len_split_text != len_lines:
        return f"Number of lines mismatched. Record " f"`text` had {len_split_text} lines, and `lines` had {len_lines}."

    for i, (text_line, line) in enumerate(zip(split_text, lines)):
        j = 0
        for word_dict in line:
            word = word_dict["word"]
            if word.strip() == "":
                return f'Line {i} ("{line}") had a word that was just whitespace at' f' index {j} ("{word}")'
            try:
                start = text_line.index(word)
            except:
                return f'Line {i} ("{line}") did not contain 0-indexed word' f' number {j} ("{word}")'
            j += 1
            text_line = text_line[start + len(word) :]

    return None
