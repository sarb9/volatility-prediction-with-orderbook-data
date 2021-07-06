from typing import List, Tuple, Callable, Dict
from dataclasses import dataclass

import numpy as np


@dataclass
class Feature:
    name: str
    function: Callable
    inputs: List[str]


feature_map: Dict[str, Feature] = {}


def register_feature(name: str, inputs: List[str] = []) -> Callable:
    def decorator(func) -> Callable:
        global feature_map
        feature_map[name] = Feature(name=name, function=func, inputs=inputs)

        return func

    return decorator
