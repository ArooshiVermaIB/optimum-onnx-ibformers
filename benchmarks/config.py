from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, TypeVar, Generic, Type, List

import yaml

T = TypeVar('T')


class _ConfigRegistry(Generic[T], metaclass=ABCMeta):

    @property
    @abstractmethod
    def _config_class(self) -> Type[T]:
        pass

    @property
    @abstractmethod
    def _config_path(self) -> Path:
        pass

    def __init__(self):
        self._registry: Dict[str, T] = dict()
        self._build_registry()

    def _build_registry(self):
        raw_content = yaml.load(self._config_path.read_text(), Loader=yaml.FullLoader)
        for id, config in raw_content.items():
            self._registry[id] = self._config_class(**config)

    def get_config(self, id_: str) -> T:
        config = self._registry.get(id_, None)
        if config is None:
            raise KeyError(f'No such configuration as {id_}')
        return config

    @property
    def available_configs(self) -> List[str]:
        return [k for k in self._registry.keys() if not k.startswith('_')]


@dataclass
class BenchmarkConfig:
    name: str
    shared_path: Path


class _BenchmarkRegistry(_ConfigRegistry[BenchmarkConfig]):
    @property
    def _config_class(self) -> Type[T]:
        return BenchmarkConfig

    @property
    def _config_path(self) -> Path:
        return Path(__file__).absolute().parent / 'configs' / 'benchmarks.yaml'


@dataclass
class ModelParamConfig:
    name: str
    hyperparams: Path


class _ModelParamsRegistry(_ConfigRegistry[ModelParamConfig]):
    DEFAULT_CONFIG_ID = '_default'

    @property
    def _config_class(self) -> Type[ModelParamConfig]:
        return ModelParamConfig

    @property
    def _config_path(self) -> Path:
        return Path(__file__).absolute().parent / 'configs' / 'model_params.yaml'

    def get_config(self, id_: str) -> ModelParamConfig:
        try:
            return super().get_config(id_)
        except KeyError:
            return super().get_config(self.DEFAULT_CONFIG_ID)


BENCHMARKS_REGISTRY = _BenchmarkRegistry()
MODEL_PARAMS_REGISTRY = _ModelParamsRegistry()
