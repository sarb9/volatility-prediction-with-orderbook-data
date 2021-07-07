import pytest

from models.nn.labeled_dataset import LabeledDataset
from timeseries.timeseries_dataset import TimeseriesDataset
from data_service.orderbook_dataset import OrderbookDataset
from timeseries.feature_extractors.features import register_feature


from tests.data_service.test_orderbook_dataset import orderbook_dataset_file_100


@pytest.fixture("module")
def create_labled_dataset_100(
    orderbook_dataset_file_100: OrderbookDataset,
):
    timeseries_dataset: TimeseriesDataset = TimeseriesDataset()
    # TODO: This line override orderbook key in feature map... has to be removed.
    from timeseries.feature_extractors.price_features import (
        extract_mid_price,
        extract_price_change,
        extract_volatility_60,
    )
    from timeseries.feature_extractors.orderbook_features import (
        extract_bid_volume,
        extract_ask_volume,
    )

    @register_feature("orderbook")
    def test_orderbook_feaurtes():
        return orderbook_dataset_file_100.books

    timeseries_dataset.add_features(
        [
            "orderbook",
            "mid_price",
            "price_change",
            "volatility_60",
            "bid_volume",
            "ask_volume",
        ]
    )

    assert timeseries_dataset.data.shape == (100, 405)

    labeled_dataset: LabeledDataset = LabeledDataset(
        timeseries_dataset=timeseries_dataset,
        features=[
            "mid_price",
            "price_change",
            "volatility_60",
            "bid_volume",
            "ask_volume",
        ],
        label="volatility_60",
        input_width=60,
        label_width=1,
        shift=1,
        train_portion=0.8,
        validation_portion=0.1,
        test_portion=0.1,
    )
