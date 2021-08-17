from typing import Dict, List
from numpy.core.fromnumeric import mean
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from timeseries.timeseries_dataset import TimeseriesDataset
from prediction_models.model import PredictionModel, import_models


EXPERIMENTS_DIRECTORY = "experiments"
EXPERIMENTS_FILE_EXTENSION = ".yaml"

KEYWORD_FEATURES = "features"

KEYWORD_HYPER_PARAMETERS = "hyper-parameters"
KEYWORD_HYPER_PARAMETERS_TRAINING_SPLIT = "training-split"
KEYWORD_HYPER_PARAMETERS_TESTING_SPLIT = "testing-split"

KEYWORD_MODELS = "models"
KEYWORD_MODEL_MODEL_NAME = "model"
KEYWORD_MODEL_NAME = "name"
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
        print("------TIMESERIES-GENERATED--------")
        print("----------------------------------")

        hyper_parameters: Dict = experiment_info[KEYWORD_HYPER_PARAMETERS]

        length: int = len(timeseries_dataset)
        training_length = int(
            hyper_parameters[KEYWORD_HYPER_PARAMETERS_TRAINING_SPLIT] * length
        )
        testing_length = int(
            hyper_parameters[KEYWORD_HYPER_PARAMETERS_TESTING_SPLIT] * length
        )

        training_split = (0, training_length)
        testing_split = (training_length, training_length + testing_length)

        training_data: TimeseriesDataset = timeseries_dataset.new_timeseries(
            split_data=training_split
        )
        testing_data: TimeseriesDataset = timeseries_dataset.new_timeseries(
            split_data=testing_split
        )

        for model_info in experiment_info[KEYWORD_MODELS]:

            model: str = model_info[KEYWORD_MODEL_MODEL_NAME]
            name: str = model_info[KEYWORD_MODEL_NAME]
            print("--------------")
            print("model", model)
            print("name", name)
            print("--------------")

            self.prediction_models[name] = self.prediction_models_classes[model](
                name,
                training_data.new_timeseries(model_info[KEYWORD_MODEL_FEATURES]),
                training_data.new_timeseries([model_info[KEYWORD_MODEL_TARGET]]),
                **model_info[KEYWORD_MODEL_CONSTRUCTION_ARGS],
            )

            print(f"Model {name} of type{model} training...")
            self.prediction_models[name].fit(**model_info[KEYWORD_MODEL_FIT_ARGS])
            print(f"Model {name} of type{model} training done.")
            # self.prediction_models[name].plot_prediction_series()

        # prediction
        y = testing_data.new_timeseries([model_info[KEYWORD_MODEL_TARGET]]).data[:, 0]
        y = (y - np.mean(y)) / np.std(y)

        legends = ["actual"]
        predictions = y
        for model_info in experiment_info[KEYWORD_MODELS]:
            name: str = model_info[KEYWORD_MODEL_NAME]

            prediction_series: np.array = self.prediction_models[name].predict(
                testing_data.new_timeseries(model_info[KEYWORD_MODEL_FEATURES]),
                testing_data.new_timeseries([model_info[KEYWORD_MODEL_TARGET]]),
            )

            diff = len(testing_data) - len(prediction_series)
            prediction: np.array = np.concatenate((np.zeros(diff), prediction_series))
            predictions = np.vstack((predictions, prediction))
            legends.append(name)

        from user_interface.main import plot_results, plot_losses

        plot_results(predictions, legends)

        plot_losses(predictions, legends)

        import pdb

        pdb.set_trace()
