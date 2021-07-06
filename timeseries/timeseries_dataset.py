import pdb
import typing
from dataclasses import dataclass

import numpy as np
from numpy.core.fromnumeric import shape

from settings import TIMESERIES_FEATURE_EXTRACTION_FEATURES

# TODO: remove the line below
from timeseries.feature_extractors.features import Feature, feature_map


class TimeseriesDataset:
    @dataclass
    class FeatureInfo:
        name: str
        slice: slice
        shape: typing.Tuple

    def __init__(self) -> None:
        self.data = np.zeros((0, 0))

        self.feature_info_map: typing.Dict[str, TimeseriesDataset.FeatureInfo] = {}

    def add_features(
        self,
        feature_names: typing.List = TIMESERIES_FEATURE_EXTRACTION_FEATURES,
    ) -> None:
        for feature_name in feature_names:
            global feature_map
            if feature_name not in feature_map.keys():
                raise Exception(
                    f"Feature {feature_name} has not been defined but it is listed in feature_names {feature_names}."
                )
            feature: Feature = feature_map[feature_name]

            inputs: typing.List[np.ndarray] = []
            for input_name in feature.inputs:
                if input_name not in self.feature_info_map:
                    raise Exception(
                        f"Feature {input_name} wanted by feature {feature_name} but it has not yet added to features {self.feature_info_map}."
                    )
                input_info: TimeseriesDataset.FeatureInfo = self.feature_info_map[
                    input_name
                ]
                input_array: np.ndarray = self.data[:, input_info.slice].reshape(
                    input_info.shape
                )

                inputs.append(input_array)

            columns: np.ndarray = feature.function(*inputs)
            actual_shape: np.shape = columns.shape
            columns = columns.reshape(columns.shape[0], -1)

            slice_start, slice_end = self.add_columns(columns)

            feature_info: TimeseriesDataset.FeatureInfo = TimeseriesDataset.FeatureInfo(
                name=feature_name,
                slice=slice(slice_start, slice_end),
                shape=actual_shape,
            )

            self.feature_info_map[feature_name] = feature_info

    def add_columns(self, columns: np.array) -> typing.Tuple[int, int]:
        if not self.data.size:
            self.data = columns
            return 0, self.data.shape[1]

        if columns.shape[0] != len(self):
            raise Exception(
                f"Trying to add array with shape {columns.shape} to {self.__class__} with length {len(self)}."
            )

        if len(columns.shape) not in {1, 2}:
            raise Exception(
                f"Trying to add non 1D or 2D array with shape {columns.shape} to {self.__class__} with length {len(self)}."
            )
        slice_start: int = self.data.shape[1]
        self.data = np.concatenate((self.data, columns), axis=1)

        return slice_start, self.data.shape[1]

    def numpy_array(self, features: typing.List[str]) -> np.ndarray:
        return np.hstack(
            [
                self.data[:, self.feature_info_map[feature_name].slice]
                for feature_name in features
            ]
        )

    def __len__(self):
        return self.data.shape[0]
