import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as smt


def correlation_analysis(
    data1: pd.Series,
    data2: pd.Series,
    *,
    title: str = "",
    ts_xlabel: str = "",
    ts_ylabel1: str = "",
    ts_ylabel2: str = "",
    ts_xticks: pd.Series | None = None,
    ts_xticklabels: pd.Series | None = None,
    corr_xticks: pd.Series | None = None,
    corr_xticklabels: pd.Series | None = None,
    fig_size: tuple[int, int] = (20, 15),
    ts_plot_opts1: dict = {},
    ts_plot_opts2: dict = {},
    corr_plot_opts: dict = {},
    suptitle: str = "",
):
    time = np.arange(len(data1))

    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(*fig_size)
    fig.suptitle(suptitle)

    # Time Series Plot
    axs[0].set_title(title)

    if ts_xticks is not None:
        axs[0].set_xticks(ts_xticks)

        if ts_xticklabels is not None:
            axs[0].set_xticklabels(ts_xticklabels)

    axs[0].set_xlabel(ts_xlabel)

    # Data 1 time series plot
    ts_ax1 = axs[0]
    ts_ax1.set_ylabel(ts_ylabel1)

    ts_1 = ts_ax1.plot(time, data1, **ts_plot_opts1)
    ts_ax1.scatter(time, data1)

    # Data 2 time series plot
    ts_ax2 = ts_ax1.twinx()

    ts_ax2.set_ylabel(ts_ylabel2)
    ts_2 = ts_ax2.plot(time, data2, **ts_plot_opts2)

    lines = ts_1 + ts_2
    labels = [line.get_label() for line in lines]
    ts_ax1.legend(lines, labels, loc=0)

    # Cross Correlation Plot
    corr_ax = axs[1]

    corr_ax.spines["left"].set_position("zero")
    corr_ax.spines["right"].set_position("zero")

    corr_ax.set_title(f"Correlacion Cruzada: {title}")
    corr_ax.set_xlabel("Lag")

    if corr_xticks is not None:
        corr_ax.set_xticks(corr_xticks)

        if corr_xticklabels is not None:
            corr_ax.set_xticklabels(corr_xticklabels)

    corr = smt.ccf(data1, data2, adjusted=False)
    corr_ax.bar(range(0, corr.size), corr, width=0.5, **corr_plot_opts)

    corr_ax.axhline(1 / np.sqrt(corr.size), color="gray", linestyle="--")
    corr_ax.axhline(-1 / np.sqrt(corr.size), color="gray", linestyle="--")

    plt.show()

    # Correlation Coefficient
    coef = data1.corr(data2, method="pearson")
    print("Coeficiente de Correlacion de Pearson:", coef)
