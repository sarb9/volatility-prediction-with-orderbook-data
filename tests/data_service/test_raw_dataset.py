from pathlib import Path

from data_service.orderbook_dataset import OrderbookDataset

from tests.settings import DATASET_RAW_DATA_DIRECTORY_TEST, DATASET_RAW_DATA_FILE_100


def test_read_small_file():
    dataset: OrderbookDataset = OrderbookDataset()
    file_path: Path = Path(DATASET_RAW_DATA_DIRECTORY_TEST) / DATASET_RAW_DATA_FILE_100
    dataset.load_from_file(file_path)
    assert len(dataset) == 100
    assert dataset.books.shape == (100, 2, 2, 100)
