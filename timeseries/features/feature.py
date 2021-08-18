from glob import glob
import os
from pathlib import Path
import importlib
from typing import List, Callable, Dict
from dataclasses import dataclass
from inspect import getmembers, isfunction


import numpy as np


@dataclass
class Feature:
    type: str
    function: Callable
    input_functions: List[str]


def get_registered_features() -> Dict[str, Feature]:
    features: Dict[str, Feature] = {}

    current_directory: str = os.path.dirname(__file__)
    python_files: List[str] = glob(f"{current_directory}/*.py")
    for python_file in python_files:
        module = importlib.import_module(f".{Path(python_file).stem}", __package__)
        functions = getmembers(module, isfunction)

        for function in functions:
            func_name, func = function
            if getattr(func, "is_feature", False):
                features[func.type] = Feature(func.type, func, func.input_functions)

    return features


def register_feature(type: str, input_functions: List[str] = []) -> Callable:
    def decorator(func) -> Callable:
        func.is_feature = True
        func.type = type
        func.input_functions = input_functions
        return func

    return decorator
