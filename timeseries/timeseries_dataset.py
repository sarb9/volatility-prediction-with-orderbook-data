import typing
from dataclasses import dataclass
import functools

import numpy as np

from timeseries.features.feature import get_registered_features, Feature

KEYWORD_FEATURE_FUNCTION = "function"
KEYWORD_FEATURE_NAME = "name"
KEYWORD_FEATURE_INPUTS = "inputs"
KEYWORD_FEATURE_ARGS = "args"


class TimeseriesDataset:
    @dataclass
    class FeatureInfo:
        name: str
        function: str
        slice: slice
        shape: typing.Tuple

    @functools.cached_property
    def registered_features(self) -> typing.Dict[str, Feature]:
        return get_registered_features()

    def __init__(self, data=np.zeros((0, 0))) -> None:
        self.data = data

        self.features_info: typing.Dict[str, TimeseriesDataset.FeatureInfo] = {}

    def add_features(
        self,
        features: typing.List[typing.Dict],
    ) -> None:
        for feature in features:

            feature_function_name: str = feature[KEYWORD_FEATURE_FUNCTION]
            if feature_function_name not in self.registered_features.keys():
                raise Exception(
                    f"Feature function {feature_function_name} is not registered. Registered features: {self.registered_features}"
                )
            feature_func: Feature = self.registered_features[feature_function_name]

            inputs_args: typing.List[np.ndarray] = []
            for input_position, input_name in enumerate(
                feature[KEYWORD_FEATURE_INPUTS]
            ):
                if input_name not in self.features_info:
                    raise Exception(
                        f"Feature {input_name} is required by {feature[KEYWORD_FEATURE_NAME]} but it has not been added."
                    )
                input_info: TimeseriesDataset.FeatureInfo = self.features_info[
                    input_name
                ]

                if input_info.function != feature_func.input_functions[input_position]:
                    raise Exception(
                        f"""Feature {feature}'s {input_position} which is {input_name},
                        is of type of {input_info.function}, but it has to be of type
                        {feature_func.input_functions[input_position]}"""
                    )

                input_arg: np.ndarray = self.data[:, input_info.slice].reshape(
                    input_info.shape
                )

                inputs_args.append(input_arg)

            columns: np.ndarray = feature_func.function(
                *inputs_args, **feature.get(KEYWORD_FEATURE_ARGS, {})
            )
            actual_shape: np.shape = columns.shape
            columns = columns.reshape(columns.shape[0], -1)
            slice_start, slice_end = self.add_columns(columns)

            feature_info: TimeseriesDataset.FeatureInfo = TimeseriesDataset.FeatureInfo(
                name=feature[KEYWORD_FEATURE_NAME],
                function=feature[KEYWORD_FEATURE_FUNCTION],
                slice=slice(slice_start, slice_end),
                shape=actual_shape,
            )
            self.features_info[feature[KEYWORD_FEATURE_NAME]] = feature_info

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

    def numpy_array(self, features: typing.List[str] = None) -> np.ndarray:
        if not features:
            return self.data
        return np.hstack(
            [
                self.data[:, self.features_info[feature_name].slice]
                for feature_name in features
            ]
        )

    def new_timeseries(self, features: typing.List[str]) -> typing.Any:
        data: np.ndarray = self.numpy_array(features)
        timeseries_dataset: TimeseriesDataset = TimeseriesDataset(data=data)
        timeseries_dataset.features_info = {
            name: feature
            for name, feature in self.features_info.items()
            if name in features
        }
        return timeseries_dataset

    def __len__(self):
        return self.data.shape[0]
