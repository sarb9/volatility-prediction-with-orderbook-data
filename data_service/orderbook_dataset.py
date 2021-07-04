from pathlib import Path
import numpy as np

from settings import DATASET_RAW_DATA_DIRECTORY, DATASET_RAW_DATA_FILE
from data_service.file_utils.raw_dataset import load_from_file


class OrderbookDataset:
    def __init__(
        self,
    ) -> None:
        self.books = np.zeros((1, 2, 2, 1), dtype=np.float32)
        self._len = 0

    def __len__(self):
        return self._len

    def load_from_file(
        self,
        file: Path = Path(DATASET_RAW_DATA_DIRECTORY) / DATASET_RAW_DATA_FILE,
    ):
        self.books, self._len = load_from_file(file)
