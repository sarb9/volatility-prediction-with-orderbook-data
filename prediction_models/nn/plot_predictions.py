import matplotlib as mpl
import matplotlib.pyplot as plt


def plot(self, model=None, max_subplots=4, plot_col=0):
    inputs, labels = self.example

    mpl.rcParams["axes.grid"] = True
    plt.figure(figsize=(12, 8))

    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f"{plot_col} [normed]")

        plt.plot(
            self.input_indices,
            inputs[n, :, plot_col],
            label="Inputs",
            marker=".",
            zorder=-10,
        )

        self.label_columns = None
        if self.label_columns:
            label_col = None if plot_col not in self.label_columns else plot_col
        else:
            label_col = plot_col

        if label_col is None:
            continue

        plt.scatter(
            self.label_indices,
            labels[n, :, label_col],
            edgecolors="k",
            label="Labels",
            c="#2ca02c",
            s=64,
        )

        if model is not None:
            predictions = model(inputs)
            plt.scatter(
                self.label_indices,
                predictions[n, :, label_col],
                marker="X",
                edgecolors="k",
                label="Predictions",
                c="#ff7f0e",
                s=64,
            )

        if n == 0:
            plt.legend()

    plt.xlabel("Time [h]")
    mpl.rcParams["axes.grid"] = False
    plt.show()
