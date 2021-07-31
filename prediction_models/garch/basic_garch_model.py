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

        import pdb; pdb.set_trace()
        self.train_data: np.ndarray = features.numpy_array()[:, 0]
        self.target_series: np.ndarray = target.numpy_array()[:, 0]
        self.test_size = int(self.train_data.shape[0] * 0.1)

    def fit(self, epochs=None, patience=None) -> None:
        print("fitting")

    def plot_prediction_series(self, subplots=3, length=100):

        rolling_predictions = []
        for i in range(self.test_size):
            train = self.train_data[:-(self.test_size-i)]
            model = arch_model(train, p=2, q=2)
            model_fit = model.fit(disp='off')
            pred = model_fit.forecast(horizon=1)
            import pdb; pdb.set_trace()
            rolling_predictions.append(pred)
            # rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

        plt.figure(figsize=(10,4))
        true, = plt.plot(self.target_series[-self.test_size:])
        preds, = plt.plot(rolling_predictions)
        test, = plt.plot(self.train_data[-self.test_size:])
        plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
        plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
        import pdb; pdb.set_trace()
        print("plottttt")

    def predict(self, features, **kwargs) -> np.ndarray:
        return np.zeros((1,1))