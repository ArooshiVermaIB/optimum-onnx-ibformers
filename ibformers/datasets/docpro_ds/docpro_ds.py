import logging
import os
import traceback
import urllib
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional, Callable, Sequence

import datasets
import numpy as np
from datasets import BuilderConfig, Features, config
from datasets.fingerprint import Hasher


from instabase.dataset_utils.shared_types import ExtractionFieldDict
from instabase.ocr.client.libs.ibocr import (
    IBOCRRecordLayout,
)
from instabase.ocr.client.libs.ocr_types import WordPolyDict
from more_itertools import consecutive_groups
from typing_extensions import TypedDict

from ibformers.data.utils import ImageProcessor

_DESCRIPTION = """\
Internal Instabase Dataset format organized into set of IbDoc files.
"""


def get_open_fn(ibsdk):
    return open if ibsdk is None else ibsdk.ibopen


# Define some helper classes for easier typing
BoundingBox = Tuple[float, float, float, float]
Span = Tuple[int, int]


class LabelEntity(TypedDict):
    name: str
    order_id: int
    text: str
    char_spans: List[Span]
    token_spans: List[Span]
    token_label_id: int


def load_datasets(dataset_paths, ibsdk):
    # use lazy imports as in model service sdk is not available
    try:
        from instabase.dataset_utils.sdk import RemoteDatasetSDK, LocalDatasetSDK
    except ImportError as err:
        logging.error(f'SDK not found: {err}')
    assert isinstance(dataset_paths, list)

    file_client = ibsdk.file_client
    username = ibsdk.username
    try:
        # load from doc pro
        if file_client is None:
            datasets_list = [LocalDatasetSDK(dataset_path) for dataset_path in dataset_paths]
        else:
            datasets_list = [
                RemoteDatasetSDK(dataset_path, file_client, username)
                for dataset_path in dataset_paths
            ]

    except Exception as e:
        logging.error(traceback.format_exc())
        raise RuntimeError(f"Error while compiling the datasets: {e}") from e

    return datasets_list


def process_labels_from_annotation(
    words: List[WordPolyDict],
    annotations: Optional[List[ExtractionFieldDict]] = None,
    label2id: Optional[Dict[str, int]] = None,
    label2ann_label_id: Optional[Dict[str, str]] = None,
    position_map: Optional[Dict] = None,
) -> Tuple[List[LabelEntity], Sequence[int]]:
    """
    Process annotations to the format expected by arrow writer
    :param words: List of words
    :param annotations: List of ExtractionFieldDict for give record
    :param label2ann_label_id: mapping from entity name to entity id used in annotation object
    :param label2id: dictionary which contain mapping from the entity name to class_id
    :param position_map: maps tuple of word_line_idx, word_in_line_idx to the global index of word
    :return: List of label entities, token_label_ids
    """

    token_label_ids = np.zeros((len(words)), dtype=np.int64)
    entities = []

    for label_name, lab_id in label2id.items():
        if lab_id == 0 and label_name == "O":
            continue

        ann_label_id = label2ann_label_id[label_name]

        entity_annotations = [ann for ann in annotations if ann['id'] == ann_label_id]
        if len(entity_annotations) > 1:
            raise ValueError('More than one ExtractionFieldDict for given entity')

        if len(entity_annotations) == 0:
            # add empty entity
            entity: LabelEntity = LabelEntity(
                name=label_name,
                order_id=0,
                text="",
                char_spans=[],
                token_spans=[],
                token_label_id=lab_id,
            )

        else:
            extraction_field_ann = entity_annotations[0]['annotations']
            if len(extraction_field_ann) > 1:
                # raise error as multi item annotations need to be supported by modelling part
                raise ValueError("Mulitple item annotations are not supported yet")

            for order_id, extraction_ann_dict in enumerate(extraction_field_ann):
                value = extraction_ann_dict['value']
                label_indexes = []
                for idx_word in extraction_ann_dict.get('words', []):
                    # get global position
                    word_id_global = position_map.get(
                        (idx_word['line_index'], idx_word['word_index'])
                    )
                    if word_id_global is None:
                        raise ValueError(f'Cannot find indexed_word in the document - {idx_word}')

                    label_indexes.append(word_id_global)

                label_indexes.sort()
                label_groups = [list(group) for group in consecutive_groups(label_indexes)]
                # create spans for groups, span will be created by getting id for first and last word in the group
                label_token_spans = [[group[0], group[-1] + 1] for group in label_groups]

                for span in label_token_spans:
                    token_label_ids[span[0] : span[1]] = lab_id

                entity: LabelEntity = LabelEntity(
                    name=label_name,
                    order_id=0,
                    text=value,
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
    # TODO: support multi-page documents, currently quite difficult in hf/datasets
    if len(page_nums) > 1:
        logging.error(
            f"Only support image modality for single-page documents. Got {len(page_nums)} pages for {ocr_path}"
        )

    lay = layouts[page_nums[0]]
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

    # img_arr_all = np.stack(img_lst, axis=0)
    return img_arr


class DocProBuilderConfig(BuilderConfig):
    """Base class for :class:`DatasetBuilder` data configuration.
    Copied from BuilderConfig class
    Contains modifications which create custom config_id (fingerprint) based on the dataset.json content
    """

    def __init__(self, *args, **kwargs):
        super(DocProBuilderConfig, self).__init__(*args, **kwargs)

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
            if len(config_id) > config.MAX_DATASET_CONFIG_ID_READABLE_LENGTH:
                config_id = self.name + "-" + Hasher.hash(suffix)
            return config_id
        else:
            return self.name


class DocProConfig(DocProBuilderConfig):
    """
    Config for Instabase Datasets which contain additional attributes related to Doc Pro dataset
    :param use_image: whether to load the image of the page
    :param ibsdk: if images is used ibsdk is used to pick up the image
    :param id2label: label id to label name mapping
    :param extraction_class_name: name of the extracted class
    :param kwargs: keyword arguments forwarded to super.
    """

    def __init__(
        self, use_image=False, ibsdk=None, id2label=None, extraction_class_name=None, **kwargs
    ):
        super(DocProConfig, self).__init__(**kwargs)
        self.use_image = use_image
        self.ibsdk = ibsdk
        self.id2label = id2label
        self.extraction_class_name = extraction_class_name


def get_docpro_ds_split(anno: Optional[Dict]):
    """
    :param anno: annotation dictionary
    :return: Tuple of boolean indictating whether file is marked as test file and split information
    """
    if anno is None:
        return False, "test"
    elif anno['is_test_file']:
        # instabase doesn't support yet separation of val and test sets.
        # TODO: we need to change that to have separate labeled sets for val and test
        return True, "val+test"
    else:
        return False, "train"


def validate_bboxes(bbox_arr, size_per_token, word_pages_arr, page_bboxes):
    for dim in range(1):
        tokens_outside_dim = np.nonzero(bbox_arr[:, 2 + dim] > size_per_token[:, dim])
        if len(tokens_outside_dim[0]) > 0:
            example_idx = tokens_outside_dim[0][0]
            ex_bbox = bbox_arr[example_idx]
            ex_page = page_bboxes[word_pages_arr[example_idx]]
            raise ValueError(f'found bbox {ex_bbox} outside of the page. bbox {ex_page}')


class DocProDs(datasets.GeneratorBasedBuilder):
    """
    Instabase internal dataset format, creation of dataset can be done by passing list of datasets
    """

    BUILDER_CONFIGS = [
        DocProConfig(
            name="docpro_ds",
            version=datasets.Version("1.0.0"),
            description="Instabase Format Datasets",
        ),
    ]

    def __init__(self, *args, **kwargs):
        super(DocProDs, self).__init__(*args, **kwargs)
        self.image_processor = (
            ImageProcessor(do_resize=True, size=224) if self.config.use_image else None
        )
        self.extraction_class_name = self.config.extraction_class_name

    def get_class_id(self, dataset_classes):
        matching_class_ids = [
            class_id
            for class_id, class_def in dataset_classes.items()
            if class_def['name'] == self.config.extraction_class_name
        ]
        if len(matching_class_ids) == 0:
            raise ValueError('extraction_class_name not found in dataset')
        return matching_class_ids[0]

    def _info(self):
        data_files = self.config.data_files
        assert isinstance(data_files, dict), "data_files argument should be a dict for this dataset"
        if "train" in data_files:
            datasets_list = load_datasets(data_files["train"], self.config.ibsdk)
            dataset_classes = datasets_list[0].metadata['classes_spec']['classes']
            class_id = self.get_class_id(dataset_classes)
            schema = dataset_classes[class_id]['schema']
            classes = ["O"] + [lab["name"] for lab in schema]
        elif "test" in data_files:
            # inference input is a list of parsedibocr files
            assert (
                self.config.id2label is not None
            ), "Need to pass directly infromation about labels for the inference"
            classes = [self.config.id2label[i] for i in range(len(self.config.id2label))]
            if classes[0] != "O":
                raise logging.error(
                    f"loaded classes does not have required format. No O class: {classes}"
                )
        else:
            raise ValueError("data_file argument should be either in train or test mode")

        ds_features = {
            "id": datasets.Value("string"),
            "split": datasets.Value("string"),
            # to distiguish test files marked in the doc pro app from all files for which predictions are needed
            "is_test_file": datasets.Value("bool"),
            # TODO: remove this column once Dataset SDK will allow for test files iterators
            "words": datasets.Sequence(datasets.Value("string")),
            "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=4)),
            # needed to generate prediction file, after evaluation
            "word_original_bboxes": datasets.Sequence(
                datasets.Sequence(datasets.Value("float32"), length=4)
            ),
            "word_page_nums": datasets.Sequence(datasets.Value("int32")),
            "word_line_idx": datasets.Sequence(datasets.Value("int32")),
            "word_in_line_idx": datasets.Sequence(datasets.Value("int32")),
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
            ds_features["images"] = datasets.Array3D(shape=(3, 224, 224), dtype="uint8")

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=datasets.Features(ds_features),
            supervised_keys=None,
        )

    def _get_annotation_generator(self, datasets_list):
        logging.info(
            f"Reading from Instabase datasets, extracting class name: {self.extraction_class_name}"
        )

        for dataset in datasets_list:
            # first determine the class name's corresponding id.
            dataset_classes = dataset.metadata['classes_spec']['classes']
            class_id = self.get_class_id(dataset_classes)
            dataset_id = dataset.metadata['id']

            # then get all records with this class id and their annotations
            class_schema = dataset_classes[class_id]['schema']
            label2ann_label_id = {field['name']: field['id'] for field in class_schema}

            for record_anno in dataset.iterator_over_annotations():
                yield record_anno, label2ann_label_id, dataset_id, class_id

    def get_annotation_from_model_service(self, records):
        # get similar format to the one defined by dataset SDK
        # Produce dummy AnnotationItem
        # TODO: import AnnotationItem type once model-service will include dataset sdk
        for record in records:
            # generate ann item: (full_path, record_index, record, anno)
            annotation_item = (record.get_document_path(), 0, record, None)
            label2ann_label_id = {lab: None for lab in self.config.id2label.values()}
            # yield annotation_item, label2ann_label_id, dataset_id, class_id
            yield annotation_item, label2ann_label_id, None, None

    def process_annotation_item(
        self,
        ann_item: Dict,
        dataset_id: str,
        class_id: str,
        label2ann_label_id: Dict,
        label2id: Dict,
    ):
        """
        process annotation_item feeded by Dataset SDK into dictionary yielded directly to Arrow writer
        :param ann_item: AnnotationItem object return by Dataset SDK or created from model service request
        :param dataset_id: id of the dataset of the processed record
        :param class_id: id of the extraction class for given dataset
        :param label2ann_label_id: mapping from entity name to entity id used in annotation object
        :param label2id: dictionary which contain mapping from the entity name to class_id
        :return:
        """
        full_path, record_index, record, anno = ann_item
        logging.info('----------------------------------')
        logging.info(full_path)
        logging.info(record_index)

        is_test_file, split = get_docpro_ds_split(anno)
        if anno is None:
            anno = {}
        anno_fields = anno.get('fields', [])
        annotated_class_id = anno.get('annotated_class_id')
        # We do include docs that dont have annotations, but we still only include
        # those that are at least of the correct class
        # The logic for including docs is as follows:
        #   - If it is of the right class and has annotations, include it
        #   - If it is not the right class, do not include it
        #   - If the class is a None type, we still include it, so that
        #     predictions occur later. During training, it is ignored since all
        #     annotations are None in this case.
        # TODO: get rid of this condition once Dataset SDK will be able to feed records of selected class
        if annotated_class_id is not None and annotated_class_id != class_id:
            logging.info("Skipping this document because it is marked as a different class")
            return None

        lines = record.get_lines()
        layouts = [mtd.get_layout() for mtd in record.get_metadata_list()]
        words = []
        word_global_idx = 0
        line_position_to_global_position_map = {}
        for line_idx, line in enumerate(lines):
            for word_idx, word in enumerate(line):
                word['line_index'] = line_idx
                word['word_index'] = word_idx
                word['word_global_idx'] = word_global_idx
                line_position_to_global_position_map[(line_idx, word_idx)] = word_global_idx
                words.append(word)
                word_global_idx += 1

        # get content of the WordPolys
        word_lst: List[str] = [w["word"] for w in words]
        bbox_arr = np.array([[w["start_x"], w["start_y"], w["end_x"], w["end_y"]] for w in words])
        word_pages_arr = np.array([w["page"] for w in words])
        word_line_idx_arr = np.array([w["line_index"] for w in words])
        word_in_line_idx_arr = np.array([w["word_index"] for w in words])

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
        size_per_token = np.take(page_bboxes[:, 2:], word_pages_arr, axis=0)
        norm_bboxes = bbox_arr * 1000 / size_per_token[:, 0:1]
        norm_page_bboxes = page_bboxes * 1000 / page_bboxes[:, 2:3]

        # Validate bboxes
        validate_bboxes(bbox_arr, size_per_token, word_pages_arr, page_bboxes)

        entities, token_label_ids = process_labels_from_annotation(
            annotations=anno_fields,
            words=words,
            label2id=label2id,
            label2ann_label_id=label2ann_label_id,
            position_map=line_position_to_global_position_map,
        )

        doc_id = f'{dataset_id}-{os.path.basename(full_path)}-{record_index}.json'

        features = {
            "id": doc_id,
            "split": split,
            "is_test_file": is_test_file,
            "words": word_lst,
            "bboxes": norm_bboxes,
            "word_original_bboxes": bbox_arr,
            "word_page_nums": word_pages_arr,
            "word_line_idx": word_line_idx_arr,
            "word_in_line_idx": word_in_line_idx_arr,
            "page_bboxes": norm_page_bboxes,
            "page_spans": page_spans,
            "token_label_ids": token_label_ids,
            "entities": entities,
        }

        if self.config.use_image:
            open_fn = get_open_fn(self.config.ibsdk)
            page_nums = list(np.unique(word_pages_arr))
            images = get_images_from_layouts(
                layouts, self.image_processor, full_path, open_fn, page_nums
            )
            # assert len(norm_page_bboxes) == len(images), "Number of images should match number of pages in document"
            features["images"] = images

        return features

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        data_files = self.config.data_files
        if "train" in data_files:
            datasets_list = load_datasets(data_files["train"], self.config.ibsdk)
            annotation_items = self._get_annotation_generator(datasets_list)

            # self.ann_label_id2label = {lab["id"]: lab["name"] for lab in annotations["labels"]}

            # Get only Train split
            # Dataset will be enriched with additional split column which will be used later to split this dataset into
            # train/val/test
            # We do it this way because current Dataset SDK require us to download the whole record
            # in order to get information about split.
            # Passing annotation_items generator for each split at this level would require to download each record
            # 3 times which will drastically increase loading time

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
            annotation_items = self.get_annotation_from_model_service(test_files)

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST, gen_kwargs={"annotation_items": annotation_items}
                ),
            ]

    def _generate_examples(self, annotation_items):
        """Yields examples."""

        label2id = self.info.features["token_label_ids"].feature._str2int
        for annotation_item, label2ann_label_id, dataset_id, class_id in annotation_items:
            doc_dict = self.process_annotation_item(
                annotation_item, dataset_id, class_id, label2ann_label_id, label2id
            )

            if doc_dict is not None:
                yield doc_dict["id"], doc_dict