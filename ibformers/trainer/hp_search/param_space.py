from dataclasses import dataclass, asdict
import json
from pathlib import Path
from transformers import is_optuna_available
from typing import Any, List, Dict, Sequence, Callable, Optional

if is_optuna_available():
    import optuna


def get_default_param_config_path():
    return Path(__file__).parent / "default_config.json"


@dataclass
class BaseParamConfig:
    name: str


@dataclass
class CategoricalParamConfig(BaseParamConfig):
    choices: List[Any]


@dataclass
class IntParamConfig(BaseParamConfig):
    low: int
    high: int
    step: int = 1
    log: bool = False


@dataclass
class FloatParamConfig(BaseParamConfig):
    low: int
    high: int
    step: Optional[int] = None
    log: bool = False


def create_param_config(json_config: Dict[str, Any]):
    json_config = json_config.copy()
    param_type = json_config.pop("type")
    if param_type == "categorical":
        return CategoricalParamConfig(**json_config)
    if param_type == "int":
        return IntParamConfig(**json_config)
    if param_type == "float":
        return FloatParamConfig(**json_config)
    raise ValueError(f"Invalid param type {param_type}")


class HyperParamSpace(Callable[["optuna.Trial"], Dict[str, Any]]):
    def __init__(self, param_configs: Sequence[BaseParamConfig]):
        param_config_mapping = {param_config.name: param_config for param_config in param_configs}
        assert len(param_config_mapping) == len(param_configs)
        self.param_config_mapping = param_config_mapping

    def is_discrete(self):
        return all([isinstance(p, CategoricalParamConfig) for p in self.param_config_mapping.values()])

    def get_grid_search_space(self):
        if not self.is_discrete():
            raise ValueError(f"Grid search config is available only for discrete param spaces.")
        return {param_name: param_config.choices for param_name, param_config in self.param_config_mapping.items()}

    @staticmethod
    def get_trial_method_for_config(trial: "optuna.Trial", config: BaseParamConfig) -> Callable:
        if isinstance(config, CategoricalParamConfig):
            return trial.suggest_categorical
        if isinstance(config, IntParamConfig):
            return trial.suggest_int
        if isinstance(config, FloatParamConfig):
            return trial.suggest_float
        raise ValueError(f"Invalid param config class {type(config)}")

    @classmethod
    def from_json_file(cls, json_path: Path):
        json_content = json.loads(json_path.read_text())
        assert isinstance(json_content, list)
        return cls.from_json_content(json_content)

    @classmethod
    def from_json_content(cls, json_content: List[Dict[str, Any]]):
        param_configs = list(map(create_param_config, json_content))
        return cls(param_configs)

    def to_json_content(self) -> List[Dict[str, Any]]:
        return [asdict(param_config) for param_config in self.param_config_mapping.values()]

    def to_json_file(self, json_path: Path):
        json_content = self.to_json_content()
        json_path.write_text(json.dumps(json_content, indent=2))

    def __call__(self, trial: "optuna.Trial") -> Dict[str, Any]:
        return {
            name: self.get_trial_method_for_config(trial, config)(**asdict(config))
            for name, config in self.param_config_mapping.items()
        }
