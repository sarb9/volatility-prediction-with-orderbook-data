from typing import Dict, List

import yaml
from pathlib import Path

from timeseries.timeseries_dataset import TimeseriesDataset


EXPERIMENTS_DIRECTORY = "experiments"
EXPERIMENTS_FILE_EXTENSION = ".yaml"

KEYWORD_FEATURES = "features"


class ExperimentEngine:
    def __init__(self) -> None:
        pass

    def run_experiment_file(self, experiment_name: str):
        file_path: str = (Path(EXPERIMENTS_DIRECTORY) / experiment_name).with_suffix(
            EXPERIMENTS_FILE_EXTENSION
        )

        experiment_info: Dict = None
        with open(file_path, "r") as stream:
            try:
                experiment_info = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        features: List = experiment_info[KEYWORD_FEATURES]

        timeseries_dataset: TimeseriesDataset = TimeseriesDataset()
        timeseries_dataset.add_features(features)
        print("----------------------------------")
        print("----------------------------------")
        print("----------------------------------")
        print("--------------DONE----------------")
        print("----------------------------------")
        print("----------------------------------")
        print("----------------------------------")
        import pdb

        pdb.set_trace()
