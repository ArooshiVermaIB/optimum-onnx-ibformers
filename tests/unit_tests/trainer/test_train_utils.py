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
        split_lst = [["train"], ["validation"], ["test"]]
        dataset_mock.keys.return_value = ["train"]
        train_ds_mock.features = ["split"]
        dataset_mock.__getitem__.side_effect = dict(train=train_ds_mock).__getitem__
        train_ds_mock.__getitem__.side_effect = dict(split=split_lst).__getitem__

        # then
        result: DatasetDict = train_utils.split_train_with_column(dataset_mock)

        # verify
        self.assertIn("train", result)
        self.assertIn("validation", result)
        self.assertIn("test", result)

    def test_split_train_with_column_validation_error_of_keys(self):
        # given
        dataset_mock = mock.MagicMock()
        dataset_mock.keys.return_value = ["train", "test"]

        # then
        with self.assertRaises(exceptions.ValidationError) as context:
            train_utils.split_train_with_column(dataset_mock)

        # verify
        self.assertIn("for splitting should contain only train set", str(context.exception))

    def test_split_train_with_column_validation_error_of_features(self):
        # given
        dataset_mock = mock.MagicMock()
        train_ds_mock = mock.MagicMock()
        dataset_mock.keys.return_value = ["train"]
        train_ds_mock.features = []
        dataset_mock.__getitem__.side_effect = dict(train=train_ds_mock).__getitem__

        # then
        with self.assertRaises(exceptions.ValidationError) as context:
            train_utils.split_train_with_column(dataset_mock)

        # verify
        self.assertIn("No column named split which is needed for splitting", str(context.exception))

    def test_get_split_score(self):
        # given
        doc_id = "doc_id_train"

        # then
        score = train_utils.get_split_score(doc_id)

        # verify
        self.assertGreater(score, 0)
        self.assertLess(score, 1)

    def test_assign_split(self):
        # given
        train_doc_id = "doc_id_train"
        eval_doc_id = "doc_id_eval_4"
        eval_size = 0.5

        # then
        split_train = train_utils.assign_split(train_doc_id, eval_size)
        split_eval = train_utils.assign_split(eval_doc_id, eval_size)

        # verify
        self.assertEqual(split_train, "train")
        self.assertEqual(split_eval, "validation")

    def test_split_eval_from_train_deterministic(self):
        # given
        train_ds_mock = mock.MagicMock()
        split_lst = ["train", "train", "train", "train", "validation+test", "test"]
        id_lst = [f"id_{i}" for i in range(len(split_lst))]
        train_ds_mock.features = ["split", "id"]
        train_ds_mock.__getitem__.side_effect = dict(split=split_lst, id=id_lst).__getitem__
        eval_size = 0.5

        # then
        split_values = train_utils.split_eval_from_train_deterministic(train_ds_mock, eval_size)

        # verify
        expected_split_values = ["train", "train", "train", "validation", "test", "test"]
        self.assertListEqual(split_values, expected_split_values)

    def test_split_eval_from_train_semideterministic(self):
        # given
        train_ds_mock = mock.MagicMock()
        split_lst = ["train", "train", "train", "train", "validation+test", "test"]
        id_lst = [f"id_{i}" for i in range(len(split_lst))]
        train_ds_mock.features = ["split", "id"]
        train_ds_mock.__getitem__.side_effect = dict(split=split_lst, id=id_lst).__getitem__
        eval_size = 0.5

        # then
        split_values = train_utils.split_eval_from_train_semideterministic(train_ds_mock, eval_size)

        # verify
        expected_split_values = ["validation", "train", "train", "validation", "test", "test"]
        self.assertListEqual(split_values, expected_split_values)

    def test_validate_dataset_sizes(self):
        # given
        raw_datasets = {
            "train": mock.MagicMock(__len__=lambda x: 10),
            "validation": mock.MagicMock(__len__=lambda x: 5),
            "test": mock.MagicMock(__len__=lambda x: 5),
        }

        # then
        train_utils.validate_dataset_sizes(raw_datasets)

    def test_validate_dataset_sizes_invalid_train(self):
        # given
        raw_datasets = {
            "train": mock.MagicMock(__len__=lambda x: 4),
            "validation": mock.MagicMock(__len__=lambda x: 5),
            "test": mock.MagicMock(__len__=lambda x: 5),
        }

        # then
        with self.assertRaisesRegex(exceptions.ValidationError, "Dataset split train"):
            train_utils.validate_dataset_sizes(raw_datasets)

    def test_validate_dataset_sizes_invalid_val(self):
        # given
        raw_datasets = {
            "train": mock.MagicMock(__len__=lambda x: 10),
            "validation": mock.MagicMock(__len__=lambda x: 1),
            "test": mock.MagicMock(__len__=lambda x: 5),
        }

        # then
        with self.assertRaisesRegex(exceptions.ValidationError, "Dataset split validation"):
            train_utils.validate_dataset_sizes(raw_datasets)

    def test_validate_dataset_sizes_invalid_test(self):
        # given
        raw_datasets = {
            "train": mock.MagicMock(__len__=lambda x: 10),
            "validation": mock.MagicMock(__len__=lambda x: 5),
            "test": mock.MagicMock(__len__=lambda x: 1),
        }

        # then
        with self.assertRaisesRegex(exceptions.ValidationError, "Dataset split test"):
            train_utils.validate_dataset_sizes(raw_datasets)


if __name__ == "__main__":
    unittest.main()
