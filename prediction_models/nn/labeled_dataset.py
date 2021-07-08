from typing import Dict, List
from h5py._hl import dataset

import tensorflow as tf
import numpy as np
from tensorflow.python.data.ops.dataset_ops import _NumpyIterator

from timeseries.timeseries_dataset import TimeseriesDataset


class LabeledDataset:
    def __init__(
        self,
        features: TimeseriesDataset,
        target: TimeseriesDataset,
        input_width: int,
        label_width: int,
        shift: int,
        train_portion: float,
        validation_portion: float,
        test_portion: float,
    ) -> None:

        self.dataset: np.ndarray = np.hstack(
            (target.numpy_array(), features.numpy_array())
        )
        # Normalize
        import pdb

        mean = np.mean(self.dataset, axis=(0))
        std = np.std(self.dataset, axis=(0))
        self.dataset = (self.dataset - mean) / std
        pdb.set_trace()

        self.input_width: int = input_width
        self.label_width: int = label_width
        self.shift: int = shift

        assert train_portion + validation_portion + test_portion == 1

        self.train_portion: int = train_portion
        self.validation_portion: int = validation_portion
        self.test_portion: int = test_portion

        # Split dataset

        length = self.dataset.shape[0]
        train_start = 0
        validation_start: int = int(length * self.train_portion)
        test_start: int = int(length * (self.train_portion + self.validation_portion))

        self.train_data_frame: np.ndarray = self.dataset[train_start:validation_start]
        self.val_data_frame: np.ndarray = self.dataset[validation_start:test_start]
        self.test_data_frame: np.ndarray = self.dataset[test_start:]

        # Create window slices

        self.total_window_size: int = self.input_width + self.shift
        window: np.ndarray = np.arange(self.total_window_size)

        self.input_slice: slice = slice(0, self.input_width)
        self.input_indices: np.ndarray = window[self.input_slice]

        self.label_start: int = self.total_window_size - self.label_width
        self.labels_slice: slice = slice(self.label_start, None)
        self.label_indices: np.ndarray = window[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]

        labels = features[:, self.labels_slice, 0]
        labels = tf.expand_dims(labels, -1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def create_tensorflow_dataset(self, data, stride=1, batch_size=64):
        dataset: np.ndarray = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=stride,
            shuffle=True,
            batch_size=batch_size,
        )

        dataset = dataset.map(self.split_window)

        return dataset

    @property
    def train(self):
        return self.create_tensorflow_dataset(self.train_data_frame)

    @property
    def val(self):
        return self.create_tensorflow_dataset(self.val_data_frame)

    @property
    def test(self):
        return self.create_tensorflow_dataset(self.test_data_frame, stride=60)

    @property
    def plot_data(self):
        data = np.array(self.val_data_frame, dtype=np.float32)

        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=128,
        )

        dataset = dataset.map(self.split_window)

        return dataset

    @property
    def example(self):
        result = getattr(self, "_example", None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result

    def __len__(self):
        return self.dataset.shape[0]
