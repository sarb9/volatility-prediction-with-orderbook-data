import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from prediction_models.model import PredictionModel
from timeseries.timeseries_dataset import TimeseriesDataset
from prediction_models.nn.labeled_dataset import LabeledDataset


class GruModel(PredictionModel):
    name = "gru"

    def __init__(
        self,
        features: TimeseriesDataset,
        target: TimeseriesDataset,
        load=False,
    ) -> None:

        self.dataset: LabeledDataset = LabeledDataset(
            features=features,
            target=target,
            input_width=60,
            label_width=1,
            shift=1,
            train_portion=0.8,
            validation_portion=0.1,
            test_portion=0.1,
        )

        self.tfmodel = tf.keras.models.Sequential(
            [
                # Shape [batch, time, features] => [batch, time, lstm_units]
                tf.keras.layers.GRU(128, return_sequences=False),
                # Shape => [batch, time, features]
                tf.keras.layers.Dense(units=1),
            ]
        )

        if load:
            self.load_model()

    def fit(self, epochs=None, patience=None) -> None:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, mode="min"
        )

        self.tfmodel.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.metrics.RootMeanSquaredError(),
            ],
        )

        history = self.tfmodel.fit(
            self.dataset.train,
            epochs=epochs,
            validation_data=self.dataset.val,
            callbacks=[
                early_stopping,
            ],
        )
        return history

    def predict(self, features, **kwargs) -> np.ndarray:
        return self.tfmodel(input).numpy().flatten()

    def save_model(self):
        self.tfmodel.save(self.checkpoint_path)

    def load_model(self):
        self.tfmodel = tf.keras.models.load_model(self.checkpoint_path)

    def evaluate(self):
        return self.tfmodel.evaluate(self.dataset.test)

    def plot_prediction_series(self, subplots=3, length=100):
        predictions = [0] * 60
        # volatilities = self.dataset.val_data_frame[self.dataset.label_column]
        volatilities = self.dataset.val_data_frame[:, 0]

        for element in self.dataset.plot_data.as_numpy_iterator():
            input, label = element
            predictions += list(self.tfmodel(input))

        plt.figure(figsize=(12, 8))

        start = 0
        for n in range(subplots):
            plt.subplot(3, 1, n + 1)
            plt.ylabel("volatility [normed]")

            end = min(len(predictions), min(len(volatilities), start + length))
            lenn = end - start

            plt.plot(
                range(lenn),
                predictions[start:end],
                label="predictions",
                zorder=-10,
            )

            plt.plot(
                range(lenn),
                volatilities[start:end],
                label="target",
                c="#ff7f0e",
            )

            start += length
            if n == 0:
                plt.legend()

        plt.xlabel("Time [h]")
        plt.show()

    def plot_prediction_distribution(self):
        predictions = np.array([])
        for sample in self.test:
            prediction = self.model(sample).numpy()
            predictions = np.append(predictions, prediction)
        labels = np.array([])
        for label in self.test:
            labels = np.append(labels, label[1].numpy())

        plt.plot(predictions)
        plt.plot(labels)
        plt.show()

        bins = np.linspace(-1, 3, 200)
        alpha = 0.7
        plt.hist(labels, bins=bins, alpha=alpha)
        plt.hist(predictions, bins=bins, alpha=alpha)
        plt.show()
