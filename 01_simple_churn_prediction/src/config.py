from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    csv_path: str
    target_column: str


class ModelConfig(BaseModel):
    sklearn_class_path: str
    sklearn_class_params: Dict[str, Any]
    model_name: Optional[str] = None


class ExperimentConfig(BaseModel):
    experiment_name: str
    random_state: Optional[int] = 42
    data: DataConfig
    model: ModelConfig
    model_output: Optional[str] = "models/churn_model.pkl"


def load_experiment_config(path: Union[str, Path]) -> ExperimentConfig:
    """Loads a YAML configuration file and returns an experiment configuration.

    Args:
        path (Union[str, Path]): Path to configuration file.

    Returns:
        ExperimentConfig: Experiment configuration
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(**data)
