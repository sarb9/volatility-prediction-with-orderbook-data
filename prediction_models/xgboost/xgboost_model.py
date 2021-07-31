from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

from prediction_models.model import PredictionModel
from timeseries.timeseries_dataset import TimeseriesDataset

class XgBoost(PredictionModel):
    name = "xgboost"

    def __init__(
        self,
        features: TimeseriesDataset,
        target: TimeseriesDataset,
        load=False,
    ) -> None:
        print("initializing garch...")

        self.train_data: np.ndarray = features.numpy_array()[:, 0]
        self.target_series: np.ndarray = target.numpy_array()[:, 0]

        self.train_size = int(self.train_data.shape[0] * 0.9)

        self.normalize_data()

    def normalize_data(self):
        self.train_data = (self.train_data - np.mean(self.train_data)) / np.std(self.train_data)
        self.target_series = (self.target_series - np.mean(self.target_series)) / np.std(self.target_series)

    def fit(self, epochs=None, patience=None) -> None:
        print("fitting")

    def xgb_predict(self, train, target,val):
        model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
        model.fit(train, target)

        val = np.array(val).reshape(1, -1)
        pred = model.predict(val)
        return pred[0]

    def plot_prediction_series(self, subplots=3, length=100):
        predictions = []
        train = self.train_data[:-1].reshape(-1, 1)
        target = self.target_series[1:]

        for i in range(self.train_size ,target.shape[0]):
            print(i)

            pred = self.xgb_predict(train[:i, :], target[:i], train[i])

            predictions.append(pred)

        import pdb; pdb.set_trace()
        error = mean_squared_error(test[:, -1], predictions, squared=False)

        return error, test[:, -1], predictions

    def predict(self, features, **kwargs) -> np.ndarray:
        return np.zeros((1,1))