import numpy as np

from timeseries.timeseries_dataset import TimeseriesDataset


def test_timeseries_add_one_column():
    timeseries_dataset: TimeseriesDataset = TimeseriesDataset()

    test_column: np.array = np.random.rand(100, 1)
    timeseries_dataset.add_columns(test_column)

    assert len(timeseries_dataset) == 100


def test_timeseries_add_two_column():
    timeseries_dataset: TimeseriesDataset = TimeseriesDataset()

    test_column: np.array = np.random.rand(100, 1)
    timeseries_dataset.add_columns(test_column)

    assert len(timeseries_dataset) == 100

    timeseries_dataset.add_columns(test_column)
    assert len(timeseries_dataset) == 100
    assert timeseries_dataset.data.shape == (100, 2)
