import json
import os
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

import datasets
from pathlib import Path
from dataclasses import dataclass
from bs4 import BeautifulSoup, NavigableString
import logging

from ibformers.data.utils import get_tokens_spans

_DESCRIPTION = """\
websrc dataset
"""

_URL = "https://websrc-data.s3.amazonaws.com/release.zip"

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
        (bbox[0] * 1000) / width,
        (bbox[1] * 1000) / width,
        (bbox[2] * 1000) / width,
        (bbox[3] * 1000) / width,
    )


def get_divs_from_contents(contents):
    """
    Goal of this funtion is to obtain list of section from a tree structure of html
    we want to have highest possible granurality so we skipp the parent sections for which children bring
    the same amount of information about underlying text
    """
    divs = {}
    for cont in contents:
        if not hasattr(cont, "attrs"):
            continue
        tid = cont.attrs.get("tid", None)
        if tid is None:
            continue

        non_empty_ns = any([isinstance(cnt, NavigableString) and cnt.text.strip() != "" for cnt in cont.contents])
        other_tags = any([not isinstance(cnt, NavigableString) for cnt in cont.contents])
        # avoid having to broad sections by limit number of descendants
        # even though we could loose some text from the parent section

        if (len(cont.contents) == 1 or non_empty_ns) and int(tid) > 2:
            if cont.text.strip() == "":
                continue
            # remove multiple spaces
            txt = " ".join(cont.text.split())
            divs[int(cont.attrs["tid"])] = {"text": txt, "other_tags": other_tags}
        else:
            child_divs = get_divs_from_contents(cont.contents)
            if len(child_divs.keys() & divs.keys()) > 0:
                raise ValueError("child contents have the same tid")
            divs.update(child_divs)

    return divs


def get_true_elid(divs, elid, answer):
    # implement some manual rules to find true elid
    prev_elid = divs.get(elid - 1, None)
    first_try = elid
    if elid == -1:
        return None
    elif elid in divs:
        first_try = elid
    elif prev_elid is not None and prev_elid["other_tags"]:
        first_try = elid - 1
    elif divs.get(elid - 2, {"text": ""})["text"].strip()[:6] == "Posted":
        first_try = elid - 2
    elif elid - 2 == 3:
        first_try = elid - 2
    elif divs.get(elid - 3, {"text": ""})["text"].strip()[:6] in ("Genres", "Countr"):
        first_try = elid - 3

    # try to find answer in neighbouring elids
    # start with elid obtained by manual rules
    new_elid = None
    for el in [first_try, elid, elid + 1, elid - 1, elid - 2, elid + 2, elid - 3, elid + 3]:
        if el not in divs:
            continue
        if divs[el]["text"].find(answer) != -1:
            new_elid = el
            break

    if new_elid is not None:
        if divs[new_elid]["text"].find(answer) == -1:
            raise ValueError("sth went wrong")
        else:
            return new_elid
    else:
        logging.warning(f"cannot find {elid} - {answer}. Question will be skipped")
        return None


class WebSrcConfig(datasets.BuilderConfig):
    """
    Config for websrc Dataset
    """

    def __init__(self, use_image=False, **kwargs):
        """BuilderConfig for wecbsrc.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WebSrcConfig, self).__init__(**kwargs)
        self.use_image = use_image


class WebSrc(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        WebSrcConfig(name="websrc", version=datasets.Version("1.0.0"), description="WebSrc dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Array2D(shape=(None, 4), dtype="int32"),
                    "page_bboxes": datasets.Array2D(shape=(None, 4), dtype="int32"),
                    "page_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
                    "entities": datasets.Sequence(
                        {
                            "name": datasets.Value("string"),  # change to id?
                            "order_id": datasets.Value("int64"),
                            # not supported yet, annotation app need to implement it
                            "text": datasets.Value("string"),
                            "token_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
                            "token_label_id": datasets.Value("int64"),
                        }
                    ),
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
        urls_to_download = _URL
        downloaded_files = Path(dl_manager.download_and_extract(urls_to_download)) / "release"
        split_path = downloaded_files / "dataset_split.csv"
        with open(split_path, "r") as split_file:
            lines = [ln.strip().split(",") for ln in split_file]

        train_dirs = [downloaded_files / ln[0] / ("0" + ln[1])[-2:] for ln in lines if ln[3] == "train"]
        dev_dirs = [downloaded_files / ln[0] / ("0" + ln[1])[-2:] for ln in lines if ln[3] == "dev"]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"dirs": train_dirs},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"dirs": dev_dirs},
            ),
        ]

    def _generate_examples(self, dirs: List[Path]):
        """Yields examples."""

        for dir in dirs:
            assert dir.is_dir()
            examples = pd.read_csv(dir / "dataset.csv", dtype=str)
            examples["file_pref"] = examples["id"].str[:9]
            ex_group = examples.groupby("file_pref")

            for fl_pref, group in ex_group:

                html_file: Path = dir / "processed_data" / (fl_pref[2:] + ".html")
                assert html_file.is_file()
                json_file: Path = dir / "processed_data" / (fl_pref[2:] + ".json")

                parsed_html = BeautifulSoup(html_file.open("r")).body
                # pyq = PyQuery(html_file.open("r").read())
                # lx = parse(str(html_file))

                with open(json_file, "r") as fl:
                    json_dict = json.load(fl)

                png_file: Path = dir / "processed_data" / (fl_pref[2:] + ".png")
                with Image.open(png_file) as img:
                    imsize = img.size
                tag2 = json_dict["2"]["rect"]

                offx = tag2["x"]
                offy = tag2["y"]

                divs = get_divs_from_contents(parsed_html.contents)
                if len(divs) < 5:
                    logging.error(f"skipping {html_file}")
                    continue
                divs_rect = {k: json_dict[str(k)]["rect"] for k in divs}
                divs_bb = {
                    k: [r["x"] - offx, r["y"] - offy, r["x"] - offx + r["width"], r["y"] - offy + r["height"]]
                    for k, r in divs_rect.items()
                }

                bboxes, tid_spans, words = self.get_text_and_bboxes(divs, divs_bb, imsize)
                labels = self.get_labels(divs, group, tid_spans, words)

                page_bboxes = [norm_bbox([0, 0, imsize[0], imsize[1]], imsize[0])]

                bboxes_fix = self.fix_bboxes(bboxes, page_bboxes)
                page_spans = [[0, len(words)]]

                yield str(fl_pref), {
                    "id": str(fl_pref),
                    "words": words,
                    "bboxes": bboxes_fix,
                    "page_bboxes": page_bboxes,
                    "page_spans": page_spans,
                    "entities": labels,
                }

    def fix_bboxes(self, bboxes, page_bboxes):
        # fix bboxes
        bboxes = np.array(bboxes)
        bboxes_fix = bboxes.clip(min=0)
        bboxes_fix[:, [0, 2]] = bboxes[:, [0, 2]].clip(max=page_bboxes[0][2])
        bboxes_fix[:, [1, 3]] = bboxes[:, [1, 3]].clip(max=page_bboxes[0][3])
        return bboxes_fix

    def get_labels(self, divs, group, tid_spans, words):
        # shuffle all the questions which correspond to given file
        # this way we get random pharaprase of the single element -
        # question about single entity can be asked in multiple ways - we are keeping only single one
        group.sample(frac=1)
        labels = []
        already_processed = {}
        dummy_tok_lab_id = 1
        for _, row in group.iterrows():
            idx = row.element_id, row.answer_start
            if idx in already_processed:
                continue
            already_processed[idx] = True
            answer = str(row.answer).strip()

            # find div
            elid = int(row.element_id)
            true_elid = get_true_elid(divs, elid, answer)
            if true_elid is None:
                continue

            # for some reason some divs are outside of image coordinates and we skip such labels
            if true_elid not in tid_spans:
                continue

            div_span = tid_spans[true_elid]
            div_words = words[div_span[0] : div_span[1]]
            div_text = " ".join(div_words)
            div_words_len = list(map(len, div_words))
            div_word_offsets = np.cumsum(np.array([-1] + div_words_len[:-1]) + 1)

            # double check if answer can be find in the div text
            ans_start = div_text.find(answer)
            assert ans_start > -1
            answer_word_span = get_tokens_spans([(ans_start, ans_start + len(answer))], div_word_offsets)[0]
            global_span = (answer_word_span[0] + div_span[0], answer_word_span[1] + div_span[0])

            ans_check = " ".join(words[global_span[0] : global_span[1]])
            if ans_check != answer:
                logging.warning(f"found answer different than gold {ans_check} != {answer}")

            lab = {
                "name": row.question,
                "order_id": 0,
                "text": answer,
                "token_spans": [global_span],
                "token_label_id": dummy_tok_lab_id,
            }  # add dummy label id
            dummy_tok_lab_id += 1

            labels.append(lab)
        return labels

    def get_text_and_bboxes(self, divs, divs_bb, imsize):
        # get word text and bboxes
        words = []
        bboxes = []
        tid_spans = {}
        max_tid = max(divs.keys())
        for tid in range(max_tid + 1):
            if tid not in divs:
                continue
            divtxt: str = divs[tid]["text"]
            div_bb = divs_bb[tid]
            if min(div_bb) < 0:
                continue
            # remove bboxes which are significantly outside of image boundaries
            if div_bb[2] > 1.1 * imsize[0] or div_bb[3] > 1.1 * imsize[1]:
                continue
            norm_div_bb = norm_bbox(div_bb, width=imsize[0])
            wrd_txt = divtxt.split()
            wrd_len = np.array([len(w) + 1 for w in wrd_txt])
            cum_len = list(np.cumsum(wrd_len))
            div_len = cum_len[-1]
            # ugly hack to maintain some spacing between sections
            div_wdth = (norm_div_bb[2] - norm_div_bb[0]) * 0.75

            wrd_bb = [
                [
                    round(norm_div_bb[0] + (l1 / div_len) * div_wdth),
                    round(norm_div_bb[1]),
                    round(norm_div_bb[0] + (l2 / div_len) * div_wdth),
                    round(norm_div_bb[3]),
                ]
                for l1, l2 in zip([0] + cum_len[:-1], cum_len)
            ]

            tid_spans[tid] = (len(words), len(words) + len(wrd_txt))

            words.extend(wrd_txt)
            bboxes.extend(wrd_bb)
        return bboxes, tid_spans, words
