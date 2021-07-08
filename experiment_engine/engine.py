from typing import Dict, List
import typing
import yaml
from pathlib import Path

from timeseries.timeseries_dataset import TimeseriesDataset

from prediction_models.model import PredictionModel, import_models


EXPERIMENTS_DIRECTORY = "experiments"
EXPERIMENTS_FILE_EXTENSION = ".yaml"

KEYWORD_FEATURES = "features"
KEYWORD_MODELS = "models"
KEYWORD_MODEL_MODEL_NAME = "model"
KEYWORD_MODEL_NAME = "model"
KEYWORD_MODEL_FEATURES = "features"
KEYWORD_MODEL_TARGET = "target"
KEYWORD_MODEL_FIT_ARGS = "fit_args"
KEYWORD_MODEL_CONSTRUCTION_ARGS = "construction_args"


class ExperimentEngine:
    def __init__(self) -> None:
        self.prediction_models_classes: Dict[str, type] = import_models()
        self.prediction_models: Dict[str, PredictionModel] = {}

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
        print("------TIMESERIES-GENERATED--------")
        print("----------------------------------")
        print("----------------------------------")
        print("----------------------------------")

        for model_info in experiment_info[KEYWORD_MODELS]:

            model: str = model_info[KEYWORD_MODEL_MODEL_NAME]
            name: str = model_info[KEYWORD_MODEL_NAME]

            self.prediction_models[name] = self.prediction_models_classes[model](
                timeseries_dataset.new_timeseries(model_info[KEYWORD_MODEL_FEATURES]),
                timeseries_dataset.new_timeseries([model_info[KEYWORD_MODEL_TARGET]]),
                **model_info[KEYWORD_MODEL_CONSTRUCTION_ARGS]
            )

            # TRAINING
            self.prediction_models[name].fit(**model_info[KEYWORD_MODEL_FIT_ARGS])
            print("FITTED")
            self.prediction_models[name].plot_prediction_series()
