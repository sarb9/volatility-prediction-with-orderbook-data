import pytest
import numpy as np
import pandas as pd

from timeseries.timeseries_dataset import TimeseriesDataset
from data_service.orderbook_dataset import OrderbookDataset
from timeseries.feature_extractors.features import register_feature

from tests.data_service.test_orderbook_dataset import orderbook_dataset_file_100


@pytest.fixture
def timeseries_dataset():
    timeseries_dataset: TimeseriesDataset = TimeseriesDataset()

    return timeseries_dataset


def test_timeseries_add_one_column(timeseries_dataset: TimeseriesDataset):
    test_column: np.array = np.random.rand(100, 1)
    timeseries_dataset.add_columns(test_column)

    assert len(timeseries_dataset) == 100


def test_timeseries_add_two_column(timeseries_dataset: TimeseriesDataset):

    test_column: np.array = np.random.rand(100, 1)
    timeseries_dataset.add_columns(test_column)

    assert len(timeseries_dataset) == 100

    timeseries_dataset.add_columns(test_column)
    assert len(timeseries_dataset) == 100
    assert timeseries_dataset.data.shape == (100, 2)


@pytest.fixture
def dummy_orderbook(
    timeseries_dataset: TimeseriesDataset,
    orderbook_dataset_file_100: OrderbookDataset,
):
    @register_feature("orderbook")
    def test_orderbook_feaurtes():
        return orderbook_dataset_file_100.books


def test_read_orderbook(
    timeseries_dataset: TimeseriesDataset,
    dummy_orderbook,
):

    timeseries_dataset.add_features(["orderbook"])

    from timeseries.feature_extractors.features import feature_map

    # TODO: we have some serious problems here...
    # assert set(feature_map.keys()) == {"orderbook"}
    assert timeseries_dataset.data.shape == (100, 400)


def test_timeseries_with_orderbook_and_mid_price(
    timeseries_dataset: TimeseriesDataset,
    dummy_orderbook: None,
):
    # TODO: We have to import unused variable to be able to use the feature
    from timeseries.feature_extractors.price_features import extract_mid_price

    timeseries_dataset.add_features(
        [
            "orderbook",
            "mid_price",
        ]
    )

    assert timeseries_dataset.data.shape == (100, 401)


def test_timeseries_with_orderbook_and_mid_price_and_price_change(
    timeseries_dataset: TimeseriesDataset,
    dummy_orderbook: None,
):
    from timeseries.feature_extractors.price_features import (
        extract_mid_price,
        extract_price_change,
    )

    timeseries_dataset.add_features(
        [
            "orderbook",
            "mid_price",
            "price_change",
        ]
    )

    assert timeseries_dataset.data.shape == (100, 402)


def test_timeseries_with_orderbook_and_mid_price_and_price_change(
    timeseries_dataset: TimeseriesDataset,
    dummy_orderbook: None,
):
    from timeseries.feature_extractors.price_features import (
        extract_mid_price,
        extract_price_change,
    )

    timeseries_dataset.add_features(
        [
            "orderbook",
            "mid_price",
            "price_change",
        ]
    )

    assert timeseries_dataset.data.shape == (100, 402)


def test_timeseries_with_orderbook_spread_and_bid_ask_volume(
    timeseries_dataset: TimeseriesDataset,
    dummy_orderbook: None,
    orderbook_dataset_file_100: OrderbookDataset,
):
    from timeseries.feature_extractors.price_features import extract_mid_price

    timeseries_dataset.add_features(
        [
            "orderbook",
        ]
    )
    assert timeseries_dataset.data.shape == (100, 400)

    from timeseries.feature_extractors.orderbook_features import (
        extract_spread,
        extract_bid_volume,
        extract_sell_volume,
    )

    # TODO: This line override orderbook key in feature map... has to be removed.
    @register_feature("orderbook")
    def test_orderbook_feaurtes():
        return orderbook_dataset_file_100.books

    timeseries_dataset.add_features(
        [
            "mid_price",
            "spread",
            "bid_volume",
            "sell_volume",
        ]
    )

    assert timeseries_dataset.data.shape == (100, 404)


def test_timeseries_with_volatility(
    timeseries_dataset: TimeseriesDataset,
    dummy_orderbook: None,
):
    from timeseries.feature_extractors.price_features import (
        extract_mid_price,
        extract_price_change,
        extract_volatility_60,
    )

    timeseries_dataset.add_features(
        [
            "orderbook",
            "mid_price",
            "price_change",
            "volatility_60",
        ]
    )
    assert timeseries_dataset.data.shape == (100, 403)


def test_timeseries_to_numpy_array(
    timeseries_dataset: TimeseriesDataset,
):
    from timeseries.feature_extractors.price_features import (
        extract_mid_price,
        extract_price_change,
        extract_volatility_60,
    )

    timeseries_dataset.add_features(
        [
            "orderbook",
            "mid_price",
            "price_change",
            "volatility_60",
        ]
    )

    array: np.ndarray = timeseries_dataset.numpy_array(
        [
            "mid_price",
            "price_change",
            "volatility_60",
        ]
    )
    assert array.shape == (100, 3)
