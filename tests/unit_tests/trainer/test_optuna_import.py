import unittest

from unittest.mock import patch, MagicMock

orig_import = __import__


def import_mock(name, *args, **kwargs):
    if "optuna" in name:
        raise ImportError()
    if "instabase" in name:  # not available on unittest pipeline
        return MagicMock()
    return orig_import(name, *args, **kwargs)


def is_optuna_available_mock():
    return False


class TestTrainingImportsWithMissingOptuna(unittest.TestCase):
    def test_import_with_missing_optuna(self):
        with patch("builtins.__import__", side_effect=import_mock) as _, patch(
            "transformers.integrations.is_optuna_available", side_effect=is_optuna_available_mock
        ) as _, patch("transformers.is_optuna_available", side_effect=is_optuna_available_mock) as _:
            with patch("transformers.integrations.is_optuna_available", side_effect=is_optuna_available_mock):
                from ibformers.trainer import train, trainer
                from ibformers.trainer import docpro_utils
