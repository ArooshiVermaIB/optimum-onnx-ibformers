import unittest
from unittest import mock

from datasets import DatasetDict

from ibformers.trainer import train_utils
from ibformers.utils import exceptions


class TestTrainUtils(unittest.TestCase):
    def test_split_train_with_column_happy_path(self):
        # given
        dataset_mock = mock.MagicMock()
        train_ds_mock = mock.MagicMock()
        split_lst = [['train'], ['validation'], ['test']]
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


if __name__ == '__main__':
    unittest.main()
