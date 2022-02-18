import dataclasses
import sys
import unittest
from unittest.mock import patch

from transformers import TrainingArguments, HfArgumentParser

from ibformers.trainer import arguments


class TestArguments(unittest.TestCase):
    def test_get_matching_commandline_params_separate_classes(self):
        # given
        param_class_a = dataclasses.make_dataclass("A", [("a", int), ("b", int)])
        param_class_b = dataclasses.make_dataclass("B", [("c", int), ("d", int)])

        parser = HfArgumentParser((param_class_a, param_class_b))

        cli_args = [sys.argv[0], "--a", "2", "--d", "5"]

        # then
        with patch("sys.argv", cli_args):
            arg_dict = arguments.get_matching_commandline_params(parser)

        # verify
        expected = {"a": 2, "d": 5}
        self.assertDictEqual(expected, arg_dict)

    def test_get_matching_commandline_params_non_init_field(self):
        # given
        param_class_a = dataclasses.make_dataclass("A", [("a", "int", dataclasses.field(default=1, init=False))])

        parser = HfArgumentParser((param_class_a))

        cli_args = [sys.argv[0], "--a", "2"]

        # then
        with patch("sys.argv", cli_args):
            arg_dict = arguments.get_matching_commandline_params(parser)

        # verify
        expected = {}
        self.assertDictEqual(expected, arg_dict)

    def test_get_matching_commandline_params_with_training_arguments(self):
        # this is a specific test case, since TrainingArguments is badly written dataclass - it overwrites
        # some of the params in postinit, so you cannot re-initialize it with its current values.
        # It does not work with `dataclasses.replace` because of that.

        # given
        parser = HfArgumentParser((TrainingArguments,))

        cli_args = [sys.argv[0], "--num_train_epochs", "1"]

        # then
        # this shouldn't fail
        # then
        with patch("sys.argv", cli_args):
            arg_dict = arguments.get_matching_commandline_params(parser)

        # verify
        expected = {"num_train_epochs": 1}
        self.assertDictEqual(expected, arg_dict)
