import os
import abc
from pathlib import Path
from typing import Dict, List
from glob import glob
import importlib
from inspect import getmembers, isfunction

import numpy as np

MODELS_SUFFIX = "_model"


class PredictionModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, features, target, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def predict(self, features, **kwargs) -> np.ndarray:
        pass

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass


def import_models() -> Dict[str, PredictionModel]:
    python_files: List[str] = glob(f"**/*{MODELS_SUFFIX}.py", recursive=True)
    for python_file in python_files:

        file_path: Path = Path(python_file)
        parts: List[str] = file_path.parts
        module_name: Path = Path(parts[-1]).stem
        module_path: str = ".".join(parts[1:-1]) + "." + module_name

        importlib.import_module(f".{module_path}", __package__)

    models: List[PredictionModel] = PredictionModel.get_subclasses()
    return {model.name: model for model in models}
