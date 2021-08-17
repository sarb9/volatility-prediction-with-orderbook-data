import numpy as np
from sklearn.svm import SVR

from prediction_models.model import PredictionModel
from timeseries.timeseries_dataset import TimeseriesDataset


class SupportVectorMachine(PredictionModel):
    name = "svr"

    def __init__(
        self,
        name: str,
        features: TimeseriesDataset,
        target: TimeseriesDataset,
    ) -> None:
        print("initializing svr...")

        self.name = name

        self.train_series = target.numpy_array()[:, 0]

        self.train_series = (self.train_series - np.mean(self.train_series)) / np.std(
            self.train_series
        )

    def fit(self) -> None:
        print("fitting")

    def _predict_single_point(self, train, target, val):
        model = SVR(kernel="rbf", C=1e3, gamma=0.1)
        model.fit(train, target)

        val = np.array(val).reshape(1, -1)
        pred = model.predict(val)
        return pred[0]

    def predict(
        self,
        testing_data_input: TimeseriesDataset,
        testing_data_observation: TimeseriesDataset,
    ) -> np.ndarray:
        # normalize data
        test_series = testing_data_observation.numpy_array()[:, 0]
        test_series = (test_series - np.mean(test_series)) / np.std(test_series)

        # predict
        data_series = np.concatenate((self.train_series, test_series), axis=0)
        input_series = data_series[:-1].reshape(-1, 1)
        target_series = data_series[1:]

        predictions = []
        for i in range(self.train_series.shape[0], input_series.shape[0]):
            print(i)

            pred = self._predict_single_point(
                input_series[:i], target_series[:i], input_series[i]
            )

            predictions.append(pred)

        return np.array(predictions)
