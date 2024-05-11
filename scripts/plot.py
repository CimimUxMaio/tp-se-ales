import numpy as np
from numpy import poly1d, polyfit


def plot_with_regression(ax, xs, ys, *, xticks=None, xticklabels=None, xticks_opts={}, fontsize=12, reg_opts={"color": "gray"}, plot="plot", plot_opts={}):
    reg_coef = polyfit(xs, ys, 1)
    regression = poly1d(reg_coef)

    if xticklabels:
        xticks = xticks or xs
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, **xticks_opts)

    slope = reg_coef[0]
    if plot == "plot":
        ax.plot(xs, ys, **plot_opts)
    elif plot == "scatter":
        ax.scatter(xs, ys, **plot_opts)
    else:
        raise ValueError("Plot option must be either 'plot' or 'scatter'")

    reg_ys = regression(xs)
    ax.plot(xs, reg_ys, **reg_opts)
    ax.text(xs[0], reg_ys[-1], f"Pendiente: {slope:.2f} °C / año", fontsize=fontsize)
