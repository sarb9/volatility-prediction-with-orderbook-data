import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from prediction_models.model import PredictionModel
from timeseries.timeseries_dataset import TimeseriesDataset
from prediction_models.nn.labeled_dataset import LabeledDataset

KEYWORD_SAVE_MODEL_DIRECTORY = "directory"


class ConvolutionalModel(PredictionModel):
    name = "convolutional"

    def __init__(
        self,
        name: str,
        features: TimeseriesDataset,
        target: TimeseriesDataset,
        input_width=60,
        label_width=1,
        shift=1,
        train_portion=0.85,
        validation_portion=0.15,
        load_model=None,
    ) -> None:

        self.name = name

        self.dataset: LabeledDataset = LabeledDataset(
            features=features,
            target=target,
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            train_portion=train_portion,
            validation_portion=validation_portion,
            test_portion=0.0,
        )

        CONV_WIDTH: int = input_width

        if load_model:
            self.tfmodel = tf.keras.models.load_model(
                load_model[KEYWORD_SAVE_MODEL_DIRECTORY] + self.name
            )
            self.train = False
        else:
            self.tfmodel = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv1D(
                        filters=32, kernel_size=(CONV_WIDTH,), activation="relu"
                    ),
                    tf.keras.layers.Dense(units=32, activation="relu"),
                    tf.keras.layers.Dense(units=1),
                ]
            )

    def fit(
        self,
        epochs=None,
        patience=None,
        save_model=None,
    ) -> None:
        if not self.train:
            return

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

        if save_model:
            self.tfmodel.save(save_model[KEYWORD_SAVE_MODEL_DIRECTORY] + self.name)

        return history

    def predict(
        self,
        testing_data_input: TimeseriesDataset,
        testing_data_observation: TimeseriesDataset,
    ) -> np.ndarray:
        test_dataset: LabeledDataset = LabeledDataset(
            features=testing_data_input,
            target=testing_data_observation,
            input_width=60,
            label_width=1,
            shift=1,
            train_portion=0.0,
            validation_portion=0.0,
            test_portion=1.0,
        )

        predictions = []
        for element in test_dataset.test_without_shuffle.as_numpy_iterator():
            input, label = element
            predictions.append(self.tfmodel(input).numpy())

        return np.vstack([prediction for prediction in predictions]).flatten()

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
