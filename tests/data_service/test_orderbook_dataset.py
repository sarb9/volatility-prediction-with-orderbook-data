from pathlib import Path
import pytest

from data_service.orderbook_dataset import OrderbookDataset

from tests.settings import DATASET_RAW_DATA_DIRECTORY_TEST, DATASET_RAW_DATA_FILE_100


@pytest.fixture(scope="module")
def orderbook_dataset_file_100():
    orderbook_dataset: OrderbookDataset = OrderbookDataset()
    file_path: Path = Path(DATASET_RAW_DATA_DIRECTORY_TEST) / DATASET_RAW_DATA_FILE_100
    orderbook_dataset.load_from_file(file_path)

    return orderbook_dataset


def test_read_small_file(orderbook_dataset_file_100: OrderbookDataset):
    assert len(orderbook_dataset_file_100) == 100
    assert orderbook_dataset_file_100.books.shape == (100, 2, 2, 100)
