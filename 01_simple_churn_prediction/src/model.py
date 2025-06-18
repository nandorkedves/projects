import importlib
from typing import Any, Optional


class SKLearnModel:
    def __init__(
        self,
        sklearn_class_path: str,
        sklearn_class_params: dict[str, Any],
        model_name: Optional[str] = None,
    ):
        module_path, _, attr_path = sklearn_class_path.rpartition(".")
        if not module_path.startswith("sklearn."):
            module_path = f"sklearn.{module_path}"

        module = importlib.import_module(module_path)

        self.name = model_name or attr_path

        self.model = getattr(module, attr_path)(**sklearn_class_params)


model_config = {"n_estimators": 100, "random_state": 42}

SKLearnModel("sklearn.ensemble.RandomForestClassifier", model_config)
