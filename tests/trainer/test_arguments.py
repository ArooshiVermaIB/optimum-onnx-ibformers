import dataclasses
import sys
import unittest
from unittest.mock import patch

from transformers import TrainingArguments

from ibformers.trainer import arguments


class TestArguments(unittest.TestCase):
    def test_update_params_with_commandline_separate_classes(self):
        # given
        param_class_a = dataclasses.make_dataclass('A', [('a', int), ('b', int)])
        param_class_b = dataclasses.make_dataclass('B', [('c', int), ('d', int)])

        params_a = param_class_a(1, 2)
        params_b = param_class_b(3, 4)

        cli_args = [sys.argv[0], "--a", "2", "--d", "5"]

        # then
        with patch("sys.argv", cli_args):
            new_params_a, new_params_b = arguments.update_params_with_commandline((params_a, params_b))

        # verify
        self.assertIsInstance(new_params_a, param_class_a)
        self.assertIsInstance(new_params_b, param_class_b)
        self.assertEqual(new_params_a.a, 2)
        self.assertEqual(new_params_b.d, 5)

    def test_update_params_with_commandline_shared_params(self):
        # given
        param_class_a = dataclasses.make_dataclass('A', [('a', int), ('b', int)])
        param_class_b = dataclasses.make_dataclass('B', [('b', int), ('c', int)])

        params_a = param_class_a(1, 2)
        params_b = param_class_b(3, 4)

        cli_args = [sys.argv[0], "--b", "5"]

        # then
        with patch("sys.argv", cli_args):
            new_params_a, new_params_b = arguments.update_params_with_commandline((params_a, params_b))

        # verify
        self.assertEqual(new_params_a.b, 5)

    def test_update_params_with_commandline_non_init_field(self):
        # given
        param_class_a = dataclasses.make_dataclass('A', [('a', 'int', dataclasses.field(default=1, init=False))])

        params_a = param_class_a()

        cli_args = [sys.argv[0], '--a', '2']

        # then
        with patch("sys.argv", cli_args):
            (new_params_a,) = arguments.update_params_with_commandline((params_a,))

        # verify
        # we expect that the CLI param is ignored, since the field is non-init

        self.assertEqual(new_params_a.a, 1)

    def test_update_with_training_arguments(self):
        # this is a specific test case, since TrainingArguments is badly written dataclass - it overwrites
        # some of the params in postinit, so you cannot re-initialize it with its current values.
        # It does not work with `dataclasses.replace` because of that.

        # given
        training_arguments = TrainingArguments(output_dir='')

        cli_args = [sys.argv[0], '--num_train_epochs', '1']

        # then
        # this shouldn't fail
        with patch("sys.argv", cli_args):
            _ = arguments.update_params_with_commandline((training_arguments,))
