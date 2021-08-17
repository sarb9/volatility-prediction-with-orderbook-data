from arch import arch_model
import numpy as np
import matplotlib.pyplot as plt

from prediction_models.model import PredictionModel
from timeseries.timeseries_dataset import TimeseriesDataset


class BasicGarch(PredictionModel):
    name = "basic-garch"

    def __init__(
        self,
        features: TimeseriesDataset,
        target: TimeseriesDataset,
        load=False,
    ) -> None:
        print("initializing garch...")

        self.train_data: np.ndarray = features.numpy_array()[:, 0]
        self.target_series: np.ndarray = target.numpy_array()[:, 0]
        self.test_size = int(self.train_data.shape[0] * 0.1)

        self.normalize_data()

    def fit(self, epochs=None, patience=None) -> None:
        print("fitting")

    def normalize_data(self):
        self.train_data = (self.train_data - np.mean(self.train_data)) / np.std(
            self.train_data
        )
        self.target_series = (
            self.target_series - np.mean(self.target_series)
        ) / np.std(self.target_series)

    def plot_prediction_series(self, subplots=3, length=100):

        rolling_predictions = []
        for i in range(self.test_size):
            train = self.train_data[: -(self.test_size - i)]
            model = arch_model(train, p=2, q=2)
            model_fit = model.fit(disp="off")
            pred = model_fit.forecast(horizon=1)
            # rolling_predictions.append(pred)
            rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))

        import pdb

        plt.figure(figsize=(10, 4))
        (true,) = plt.plot(self.target_series[-self.test_size :])
        (preds,) = plt.plot(rolling_predictions)
        (test,) = plt.plot(self.train_data[-self.test_size :])
        plt.title("Volatility Prediction - Rolling Forecast", fontsize=20)
        plt.legend(["True Volatility", "Predicted Volatility"], fontsize=16)
        import pdb

        print("plottttt")

    def predict(self, features, **kwargs) -> np.ndarray:
        return np.zeros((1, 1))
