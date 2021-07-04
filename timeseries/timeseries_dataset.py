from typing import List
import typing

import numpy as np

from data_service.orderbook_dataset import OrderbookDataset

slices: typing.Dict[str, slice] = {}


class TimeseriesDataset:
    def __init__(self) -> None:
        self.data = np.zeros((0))

    def add_columns(self, columns: np.array):
        if not self.data.size:
            self.data = columns
            return

        if columns.shape[0] != len(self):
            raise Exception(
                f"Trying to add array with shape {columns.shape} to {self.__class__} with length {len(self)}."
            )

        if len(columns.shape) not in {1, 2}:
            raise Exception(
                f"Trying to add non 1D or 2D array with shape {columns.shape} to {self.__class__} with length {len(self)}."
            )

        self.data = np.concatenate((self.data, columns), axis=1)

    def __len__(self):
        return self.data.shape[0]
