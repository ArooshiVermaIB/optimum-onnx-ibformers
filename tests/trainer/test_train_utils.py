import dataclasses
import sys
import unittest
from unittest import mock
from unittest.mock import patch

from datasets import Dataset, DatasetDict
from ibformers.trainer import train_utils
from ibformers.utils import exceptions


class TestTrainUtils(unittest.TestCase):
    def test_split_train_with_column_happy_path(self):
        # given
        dataset_mock = mock.MagicMock()
        train_ds_mock = mock.MagicMock()
        split_lst = [['train'], ['val'], ['test']]
        dataset_mock.keys.return_value = ['train']
        train_ds_mock.features = ['split']
        dataset_mock.__getitem__.side_effect = dict(train=train_ds_mock).__getitem__
        train_ds_mock.__getitem__.side_effect = dict(split=split_lst).__getitem__

        # then
        result: DatasetDict = train_utils.split_train_with_column(dataset_mock)

        # verify
        self.assertIn('train', result)
        self.assertIn('validation', result)
        self.assertIn('test', result)

    def test_split_train_with_column_validation_error_of_keys(self):
        # given
        dataset_mock = mock.MagicMock()
        dataset_mock.keys.return_value = ['train', 'test']

        # then
        with self.assertRaises(exceptions.ValidationError) as context:
            train_utils.split_train_with_column(dataset_mock)

        # verify
        self.assertIn("for splitting should contain only train set", str(context.exception))

    def test_split_train_with_column_validation_error_of_features(self):
        # given
        dataset_mock = mock.MagicMock()
        train_ds_mock = mock.MagicMock()
        dataset_mock.keys.return_value = ['train']
        train_ds_mock.features = []
        dataset_mock.__getitem__.side_effect = dict(train=train_ds_mock).__getitem__

        # then
        with self.assertRaises(exceptions.ValidationError) as context:
            train_utils.split_train_with_column(dataset_mock)

        # verify
        self.assertIn("No column named split which is needed for splitting", str(context.exception))

    def test_update_params_with_commandline_separate_classes(self):
        # given
        param_class_a = dataclasses.make_dataclass('A', [('a', int), ('b', int)])
        param_class_b = dataclasses.make_dataclass('B', [('c', int), ('d', int)])

        params_a = param_class_a(1, 2)
        params_b = param_class_b(3, 4)

        cli_args = [sys.argv[0], "--a", "2", "--d", "5"]

        # then
        with patch("sys.argv", cli_args):
            new_params_a, new_params_b = train_utils.update_params_with_commandline((params_a, params_b))

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
            new_params_a, new_params_b = train_utils.update_params_with_commandline((params_a, params_b))

        # verify
        self.assertEqual(new_params_a.b, 5)


if __name__ == '__main__':
    unittest.main()
