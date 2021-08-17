import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np


def plot_results(predictions, legends):
    st.write(
        """
    # Volatility Prediction:
    ## Results:
    """
    )

    actual = predictions[0]
    mse = tf.keras.losses.MeanSquaredError()

    losses = mse(actual, predictions[1]).numpy()

    actual = pd.DataFrame(predictions.T, columns=legends)
    st.line_chart(actual)


def plot_losses(predictions, legends):
    st.write(
        """
        # Losses:
        """
    )

    actual = predictions[0]

    mse = tf.keras.losses.MeanSquaredError()
    mape = tf.keras.losses.MeanAbsolutePercentageError()
    mae = tf.keras.losses.MeanAbsoluteError()

    mse_losses = np.diag(
        [mse(actual, predictions[i]).numpy() for i in range(1, len(predictions))]
    )
    mape_losses = np.diag(
        [mape(actual, predictions[i]).numpy() for i in range(1, len(predictions))]
    )
    mae_losses = np.diag(
        [mae(actual, predictions[i]).numpy() for i in range(1, len(predictions))]
    )

    st.write(
        """
        ## Mean Squared Error:
        """
    )
    st.bar_chart(pd.DataFrame(mse_losses, columns=legends[1:]))
    st.write(
        """
        ## Mean Absolute Percantage Error:
        """
    )
    st.bar_chart(pd.DataFrame(mae_losses, columns=legends[1:]))
    st.write(
        """
        ## Mean Absolute Error:
        """
    )
    st.bar_chart(pd.DataFrame(mape_losses, columns=legends[1:]))
