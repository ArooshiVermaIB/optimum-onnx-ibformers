import json
import urllib
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union, NamedTuple, Optional, Callable
import datasets
from datasets import BuilderConfig, Features, config
from datasets.fingerprint import Hasher

from instabase.ocr.client.libs.ibocr import ParsedIBOCRBuilder, IBOCRRecord, IBOCRRecordLayout, ParsedIBOCR
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
    type: Literal['rect']
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

        if self.data_files is not None:
            m = Hasher()
            if suffix:
                m.update(suffix)

            # stat_fn = self.ibsdk.stat if self.ibsdk is not None else os.path.getmtime
            # stat = str(stat_fn(self.data_files["train"]))
            open_fn = get_open_fn(self.ibsdk)
            with open_fn(self.data_files["train"], 'r') as annotation_file:
                content = json.load(annotation_file)
                fingerprint_content = {k: v for k, v in content.items() if k in ('files', 'labels', 'testFiles')}
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


def process_parsedibocr(parsedibocr,
                        open_fn,
                        doc_annotations: AnnotationFile,
                        id2label: Dict[str, str],
                        str2int: Dict[str, int],
                        use_image: bool,
                        image_processor: ImageProcessor,
                        ):
    """
    prepare examples based on annotations and ocr information
    :param image_processor:
    :param doc_annotations: content of ibannotator file
    :param id2label: dictionary which contain mapping from entity ids to their names
    :param str2int: mapping from entity name to class_id used for token classification
    :param use_image: whether to load an image as a feature
    :return: Dictionary containing words, bboxes and per-word annotations
    """

    words = []
    layouts = []
    record: IBOCRRecord

    # Assuming each record is a page in order and each record is single-page
    # Assuming nothing weird is going on with page numbers
    for record in parsedibocr.get_ibocr_records():
        words += [i for j in record.get_lines() for i in j]
        l = record.get_metadata_list()
        layouts.extend([i.get_layout() for i in l])

    assert all(
        word['page'] in range(len(layouts)) for word in words
    ), "Something with the page numbers went wrong"

    # get content of the WordPollys
    word_lst = [w['word'] for w in words]
    bbox_arr = np.array([[w['start_x'], w['start_y'], w['end_x'], w['end_y']] for w in words])
    word_pages_arr = np.array([w['page'] for w in words])

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

    entities = []
    annotation: Annotation
    token_label_ids = np.zeros((len(word_lst)), dtype=np.int64)
    for label_id, annotation in doc_annotations['annotations'].items():
        metadata = annotation['metadata']
        label_name = id2label[label_id]
        word_metadata: AnnotationWordMetadata
        label_words = []
        for word_metadata in metadata:
            if word_metadata['dataType'] != 'WordPoly':
                raise ValueError("Unexpected (non-wordpoly) annotation. Skipping.")
            word_id_in_page = int(word_metadata['appData']['id']) - 1  # switch to 0-indexing
            page_num = word_metadata['position']['page']
            word_id_global = word_id_in_page + page_spans[page_num, 0]
            word_text = word_metadata['rawWord']
            assert word_text.strip() == words[word_id_global]['word'].strip(), \
                "Annotation does not match with document words"

            label_words.append({'id': word_id_global, 'text': word_text})

        # TODO: information about multi-item entity separation should be obtained during annotation
        # group consecutive words as the same entity occurrence
        label_words.sort(key=lambda x: x["id"])
        label_groups = [list(group) for group in consecutive_groups(label_words, ordering=lambda x: x["id"])]
        # create spans for groups, span will be created by getting id for first and last word in the group
        label_token_spans = [[group[0]["id"], group[-1]["id"] + 1] for group in label_groups]

        for span in label_token_spans:
            token_label_ids[span[0]:span[1]] = str2int[label_name]

        entity = {'name': label_name,
                  'order_id': 0,
                  'text': annotation['value'],
                  'char_spans': [],
                  'token_spans': label_token_spans}
        entities.append(entity)

    features = {
        "id": doc_annotations["ocrPath"],
        "words": word_lst,
        "bboxes": norm_bboxes,
        "word_original_bboxes": bbox_arr,
        "word_page_nums": word_pages_arr,
        "page_bboxes": norm_page_bboxes,
        "page_spans": page_spans,
        'token_label_ids': token_label_ids,
        "entities": entities,
    }

    if use_image:
        img_lst = []
        # TODO: support multi-page documents, currently quite difficult in hf/datasets
        for lay in layouts[:1]:
            img_path = Path(lay.get_processed_image_path())
            try:
                with open_fn(str(img_path)) as img_file:
                    img_arr = image_processor(img_file).astype(np.uint8)
            except OSError:
                # try relative path - useful for debugging
                ocr_path = Path(doc_annotations['ocrPath'])
                img_rel_path = ocr_path.parent.parent / 's1_process_files' / 'images' / img_path.name
                with open_fn(img_rel_path, 'rb') as img_file:
                    img_arr = image_processor(img_file).astype(np.uint8)

            img_lst.append(img_arr)

        # assert len(norm_page_bboxes) == len(img_lst), "Number of images should match number of pages in document"
        # features['images'] = np.stack(img_lst, axis=0)
        features['images'] = img_arr

    return features


class IbDsConfig(IbDsBuilderConfig):
    """
    Config for Instabase Format Datasets
    """

    def __init__(self, use_image=True, ibsdk=None, **kwargs):
        """BuilderConfig for Instabase Format Datasets.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(IbDsConfig, self).__init__(**kwargs)
        self.use_image = use_image
        self.ibsdk = ibsdk


class IbDs(datasets.GeneratorBasedBuilder):
    """TODO(ibds): Short description of my dataset."""

    # # Class for the builder config.
    # BUILDER_CONFIG_CLASS = IbDsBuilderConfig

    BUILDER_CONFIGS = [
        IbDsConfig(name="ibds", version=datasets.Version("1.0.0"), description="Instabase Format Datasets"),
    ]

    def __init__(self, *args, **kwargs):
        super(IbDs, self).__init__(*args, **kwargs)
        self.image_processor = ImageProcessor(do_resize=True, size=224) if self.config.use_image else None

    def _info(self):

        # get schema of the the dataset
        # TODO(ibds): Check if schema can be saved in the separate file, so we don't load whole annotation file
        data_files = self.config.data_files
        assert len(data_files) == 1, 'Only one annotation path should be provided'
        if isinstance(data_files, dict):
            data_files = list(data_files.values())
        open_fn = get_open_fn(self.config.ibsdk)
        with open_fn(data_files[0], 'r') as annotation_file:
            labels = json.load(annotation_file)["labels"]

        self.id2label = {lab["id"]: lab["name"] for lab in labels}
        classes = ['O'] + list(self.id2label.values())

        ds_features = {
                    'id': datasets.Value('string'),
                    'words': datasets.Sequence(datasets.Value('string')),
                    'bboxes': datasets.Sequence(datasets.Sequence(datasets.Value('int32'), length=4)),
                    # needed to generate prediction file, after evaluation
                    'word_original_bboxes': datasets.Sequence(datasets.Sequence(datasets.Value('float32'), length=4)),
                    'word_page_nums': datasets.Sequence(datasets.Value('int32')),
                    'page_bboxes': datasets.Sequence(datasets.Sequence(datasets.Value('int32'), length=4)),
                    'page_spans': datasets.Sequence(datasets.Sequence(datasets.Value('int32'), length=2)),
                    'token_label_ids': datasets.Sequence(datasets.features.ClassLabel(names=classes)),
                    'entities': datasets.Sequence(
                        {
                            'name': datasets.Value('string'),  # change to id?
                            'order_id': datasets.Value('int64'),  # not supported yet, annotation app need to implement it
                            'text': datasets.Value('string'),
                            'char_spans': datasets.Sequence(datasets.Sequence(datasets.Value('int32'), length=2)),
                            'token_spans': datasets.Sequence(datasets.Sequence(datasets.Value('int32'), length=2))
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
            ds_features['images'] = datasets.Array3D(shape=(3, 224, 224), dtype="uint8")

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(ds_features),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        data_files = self.config.data_files
        if isinstance(data_files, dict):
            data_files = list(data_files.values())

        open_fn = get_open_fn(self.config.ibsdk)
        with open_fn(data_files[0], 'r') as annotation_file:
            annotations = json.load(annotation_file)

        self.id2label = {lab["id"]: lab["name"] for lab in annotations["labels"]}

        # create generators for train and test
        train_files = (file for file in annotations['files']
                       if file['id'] not in annotations['testFiles'] and len(file['annotations']) > 0)
        val_files = (file for file in annotations['files']
                     if file['id'] in annotations['testFiles'] and len(file['annotations']) > 0)
        # test set is the sum of unannotated documents and validation set
        test_files = (file for file in annotations['files']
                      if len(file['annotations']) == 0 or file['id'] in annotations['testFiles'])

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={'files': train_files, 'open_fn': open_fn}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={'files': val_files, 'open_fn': open_fn}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={'files': test_files, 'open_fn': open_fn}),
        ]

    def _generate_examples(self, files, open_fn):
        """Yields examples."""
        str2int = self.info.features['token_label_ids'].feature._str2int

        for annotations in files:
            ocr_path = annotations['ocrPath']
            if not (ocr_path.endswith('.ibdoc') or ocr_path.endswith('.ibocr')):
                raise ValueError(f"Invaild document path: {ocr_path}")

            # TODO use context manager and close
            with open_fn(ocr_path, 'rb') as f:
                data = f.read()
            builder: ParsedIBOCRBuilder
            builder, err = ParsedIBOCRBuilder.load_from_str(ocr_path, data)
            if err:
                raise IOError(u'Could not load file: {}'.format(ocr_path))

            doc_dict = process_parsedibocr(builder, open_fn, annotations,
                                           self.id2label, str2int, self.config.use_image, self.image_processor)

            yield annotations['ocrPath'], doc_dict
