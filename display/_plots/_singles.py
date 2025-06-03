# Imports

# Packages
import numpy as np

from matplotlib.axes import Axes

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from skmisc.loess import loess

from scipy import stats
from scipy.stats import pearsonr

import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW

# Project
from app import reweightErrors

# --------------------------------------------------------------------

colorBlindDict = {
    'blue':    '#377eb8',
    'orange':  '#ff7f00',
    'green':   '#4daf4a',
    'pink':    '#f781bf',
    'brown':   '#a65628',
    'purple':  '#984ea3',
    'gray':    '#999999',
    'red':     '#e41a1c',
    'yellow':  '#dede00'
}
# --------------------------------------------------------------------


def plotCurveFit_annot(ax: Axes, X: np.ndarray, Y: np.ndarray):
    """
    Plots data and fits curve. Adds additional metrics to plot.

    Parameters
    ----------
    ax: axis object
        Object to be modified by the function.
    X: numpy vector
        True/independant values.
    Y: numpy vector
        Estimated/dependant values.
    """
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    bins = 10

    X, Y, weights = reweightErrors(X, Y, bins)
    E = (X-Y)*weights

    Xo = np.min(X)

    linearityMeasure = pearsonr(X, Y)
    pearsonStat = linearityMeasure.statistic
    pearsonP = linearityMeasure.pvalue
    ax.scatter(X, Y, alpha=0.5, c=colorBlindDict['blue'])
    X_hat = np.linspace(np.min(X), np.max(X), 200)

    rmse = root_mean_squared_error(X, Y, sample_weight=weights)
    mse = mean_squared_error(X, Y, sample_weight=weights)

    max_plot = max(np.max(Y), np.max(X))
    ax.annotate("MSE: {:.4f}".format(mse),  xy=(Xo, 0.95*max_plot))
    ax.annotate("RMSE: {:.4f}".format(rmse),  xy=(Xo, 0.4*max_plot))
    ax.annotate("PPMCC: {:.4f}".format(pearsonStat),  xy=(Xo, 0.7*max_plot))
    ax.annotate("PPMCC pval: {:.4f}".format(pearsonP),  xy=(Xo, 0.65*max_plot))

    # Linear fit
    linearM = LinearRegression().fit(X.reshape(-1, 1), Y.reshape(-1, 1))
    Y_hat_lin = linearM.predict(X.reshape(-1, 1))
    r2_linear = r2_score(Y, Y_hat_lin)
    ax.annotate("r-squared linear: {:.4f}".format(r2_linear),
                xy=(Xo, 0.45*max_plot))

    # Curve fit
    modelL = loess(X, Y)
    modelL.fit()
    pred = modelL.predict(X, stderror=True)
    Y_hat_loess_X = pred.values

    r2_loess = r2_score(Y, Y_hat_loess_X)

    pred = modelL.predict(X_hat, stderror=True)
    Y_hat_loess = pred.values
    conf = pred.confidence(alpha=0.01)
    ll = conf.lower
    ul = conf.upper
    Y_hat_line = pred.values

    lower = np.zeros(len(ll))
    upper = np.zeros(len(ll))

    for i in range(len(ll)):
        lower[i] = Y_hat_line[i] - stats.t.ppf(0.95, pred.df)*pred.stderr[i]
        upper[i] = Y_hat_line[i] + stats.t.ppf(0.95, pred.df)*pred.stderr[i]

    ax.plot(X_hat, Y_hat_line, c=colorBlindDict['orange'])
    ax.fill_between(X_hat.reshape(-1), lower, upper,
                    alpha=.33, color=colorBlindDict['orange'])
    ax.annotate("r-squared loess: {:.2f}".format(r2_loess),
                xy=(Xo, 0.5*max_plot))

    # error with weights
    m = DescrStatsW(X-Y, weights=weights)
    per = m.quantile([2.5/100, 97.5/100], return_pandas=False)

    ax.annotate(f"Error Bound 2.5: {round(per[0], 2)}",
                xy=(Xo, 0.90*max_plot))
    ax.annotate(f"Error Bound 97.5: {round(per[1], 2)}",
                xy=(Xo, 0.85*max_plot))

    per = np.percentile(X_hat-Y_hat_loess,
                        [2.5, 97.5], method='median_unbiased')
    ax.annotate(f"Curve Error Bound 2.5: {round(per[0],2)}",
                xy=(Xo, 0.80*max_plot))
    ax.annotate(f"Curve Error Bound 97.5: {round(per[1],2)}",
                xy=(Xo, 0.75*max_plot))


def plotCurveFit_noAnnot(ax: Axes, X: np.ndarray, Y: np.ndarray):
    """
    Plots data and fits curve. Does not display additional metrics
    on graph.

    Parameters
    ----------
    ax: axis object
        Object to be modified by the function.
    X: numpy vector
        True/independant values.
    Y: numpy vector
        Estimated/dependant values.
    """
    X = X.reshape(-1)
    Y = Y.reshape(-1)

    ax.scatter(X, Y, alpha=0.5, c=colorBlindDict['blue'], label='Estimates')
    X_hat = np.linspace(np.min(X), np.max(X), 200)

    modelL = loess(X, Y)
    modelL.fit()
    pred = modelL.predict(X_hat, stderror=True)
    Y_hat_line = pred.values

    ax.plot(X_hat, Y_hat_line, c=colorBlindDict['orange'], label='Curve Fit')


def plotLinearFit(ax: Axes, X: np.ndarray, Y: np.ndarray):
    """
    Plots data and fits line.

    Parameters
    ----------
    ax: axis object
        Object to be modified by the function.
    X: numpy vector
        True/independant values.
    Y: numpy vector
        Estimated/dependant values.
    """
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    bins = 11

    X, Y, weights = reweightErrors(X, Y, bins)
    E = X - Y
    ax.scatter(X, Y, alpha=0.5, c=colorBlindDict['blue'], label='Estimates')

    X_hat = np.linspace(np.min(X), np.max(X), 100)
    linearM = LinearRegression().fit(X.reshape(-1, 1), Y.reshape(-1, 1))
    Y_hat = linearM.predict(X.reshape(-1, 1))

    mse = mean_squared_error(X, Y)
    yR2 = r2_score(Y, Y_hat)

    max_plot = max(max(np.max(Y_hat), np.max(Y)), np.max(X))
    ax.annotate("r-squared = {:.3f}".format(yR2),
                xy=(np.min(X), 0.9*max_plot))
    ax.annotate("Linear fit slope = {:.3f}".format(linearM.coef_[0][0]),
                xy=(np.min(X), 0.85*max_plot))
    ax.annotate("MSE Bayes error = {:.3f}".format(mse),
                xy=(np.min(X), 0.75*max_plot))

    # Average error
    avge = np.average(np.abs(E))
    ax.annotate("Average error +/- {:.3f}".format(avge),
                xy=(np.min(X), 0.55*max_plot))
    # error with weights
    m = DescrStatsW(X-Y, weights=weights)
    per = m.quantile([2.5/100, 97.5/100], return_pandas=False)
    ax.annotate("Error Bounds {:.3f} {:.3f}".format(per[0], per[1]),
                xy=(np.min(X), 0.50*max_plot))

    # prediction interval
    smX = sm.add_constant(X)
    smX_hat = sm.add_constant(X_hat.reshape(-1))
    model = sm.OLS(Y, smX)
    olsObj = model.fit()

    predSumFrame = olsObj.get_prediction(smX_hat).summary_frame(alpha=0.1)

    Y_hat = predSumFrame["mean"]
    # prediction interval
    lower = predSumFrame["obs_ci_lower"]
    upper = predSumFrame["obs_ci_upper"]

    lowE = max(Y_hat-lower)
    highE = max(upper-Y_hat)
    ax.plot(X_hat, Y_hat, c=colorBlindDict['orange'], label='Linear Fit')
    ax.fill_between(X_hat.reshape(-1), lower, upper,
                    alpha=.33, color=colorBlindDict['orange'])

    ax.annotate(f"90\% prediction +/- {round(highE,3)}",
                xy=(np.min(X), 0.7*max_plot))
    predSumFrame = olsObj.get_prediction(smX_hat).summary_frame(alpha=0.05)
    predMean = predSumFrame["mean"]
    predLower = predSumFrame["obs_ci_lower"]
    hRange = max(predMean - predLower)
    ax.annotate(f"95\% prediction +/- {round(hRange,3)}",
                xy=(np.min(X), 0.65*max_plot))

    erInt = np.abs(np.average(X-Y))+2*np.std(X-Y)
    ax.annotate(f"Error 95\% +/- {round(erInt,3)}",
                xy=(np.min(X), 0.60*max_plot))
