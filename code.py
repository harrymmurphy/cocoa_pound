import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
import arch
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import deprecate_kwarg
from statsmodels.compat.python import Literal, lzip
from statsmodels.compat.scipy import _next_regular
from typing import Union
import warnings
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tools.sm_exceptions import (
    CollinearityWarning,
    InfeasibleTestError,
    InterpolationWarning,
    MissingDataError,
    ValueWarning,)
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    dict_like,
    float_like,
    int_like,
    string_like,)
from statsmodels.tsa._bds import bds
from statsmodels.tsa._innovations import innovations_algo, innovations_filter
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.tsatools import add_trend, lagmat, lagmat2ds
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson


ArrayLike1D = Union[np.ndarray, pd.Series, list[float]]

__all__ = [
    "adfuller",
    "zivot_andrews"]

SQRTEPS = np.sqrt(np.finfo(np.double).eps)


def _autolag(
    mod,
    endog,
    exog,
    startlag,
    maxlag,
    method,
    modargs=(),
    fitargs=(),
    regresults=False,
):
    """
    Returns the results for the lag length that maximizes the info criterion.

    Parameters
    ----------
    mod : Model class
        Model estimator class
    endog : array_like
        nobs array containing endogenous variable
    exog : array_like
        nobs by (startlag + maxlag) array containing lags and possibly other
        variables
    startlag : int
        The first zero-indexed column to hold a lag.  See Notes.
    maxlag : int
        The highest lag order for lag length selection.
    method : {"aic", "bic", "t-stat"}
        aic - Akaike Information Criterion
        bic - Bayes Information Criterion
        t-stat - Based on last lag
    modargs : tuple, optional
        args to pass to model.  See notes.
    fitargs : tuple, optional
        args to pass to fit.  See notes.
    regresults : bool, optional
        Flag indicating to return optional return results

    Returns
    -------
    icbest : float
        Best information criteria.
    bestlag : int
        The lag length that maximizes the information criterion.
    results : dict, optional
        Dictionary containing all estimation results

    Notes
    -----
    Does estimation like mod(endog, exog[:,:i], *modargs).fit(*fitargs)
    where i goes from lagstart to lagstart+maxlag+1.  Therefore, lags are
    assumed to be in contiguous columns from low to high lag length with
    the highest lag in the last column.
    """
    # TODO: can tcol be replaced by maxlag + 2?
    # TODO: This could be changed to laggedRHS and exog keyword arguments if
    #    this will be more general.

    results = {}
    method = method.lower()
    for lag in range(startlag, startlag + maxlag + 1):
        mod_instance = mod(endog, exog[:, :lag], *modargs)
        results[lag] = mod_instance.fit()

    if method == "aic":
        icbest, bestlag = min((v.aic, k) for k, v in results.items())
    elif method == "bic":
        icbest, bestlag = min((v.bic, k) for k, v in results.items())
    elif method == "t-stat":
        # stop = stats.norm.ppf(.95)
        stop = 1.6448536269514722
        # Default values to ensure that always set
        bestlag = startlag + maxlag
        icbest = 0.0
        for lag in range(startlag + maxlag, startlag - 1, -1):
            icbest = np.abs(results[lag].tvalues[-1])
            bestlag = lag
            if np.abs(icbest) >= stop:
                # Break for first lag with a significant t-stat
                break
    else:
        raise ValueError(f"Information Criterion {method} not understood.")

    if not regresults:
        return icbest, bestlag
    else:
        return icbest, bestlag, results

def adfuller(
    x,
    maxlag: int | None = None,
    regression="c",
    autolag="AIC",
    store=False,
    regresults=False,
):
    """
    Augmented Dickey-Fuller unit root test.

    The Augmented Dickey-Fuller test can be used to test for a unit root in a
    univariate process in the presence of serial correlation.

    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    maxlag : {None, int}
        Maximum lag which is included in test, default value of
        12*(nobs/100)^{1/4} is used when ``None``.
    regression : {"c","ct","ctt","n"}
        Constant and trend order to include in regression.

        * "c" : constant only (default).
        * "ct" : constant and trend.
        * "ctt" : constant, and linear and quadratic trend.
        * "n" : no constant, no trend.

    autolag : {"AIC", "BIC", "t-stat", None}
        Method to use when automatically determining the lag length among the
        values 0, 1, ..., maxlag.

        * If "AIC" (default) or "BIC", then the number of lags is chosen
          to minimize the corresponding information criterion.
        * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test.
        * If None, then the number of included lags is set to maxlag.
    store : bool
        If True, then a result instance is returned additionally to
        the adf statistic. Default is False.
    regresults : bool, optional
        If True, the full regression results are returned. Default is False.

    Returns
    -------
    adf : float
        The test statistic.
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994, 2010).
    usedlag : int
        The number of lags used.
    nobs : int
        The number of observations used for the ADF regression and calculation
        of the critical values.
    critical values : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels. Based on MacKinnon (2010).
    icbest : float
        The maximized information criterion if autolag is not None.
    resstore : ResultStore, optional
        A dummy class with results attached as attributes.

    Notes
    -----
    The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
    root, with the alternative that there is no unit root. If the pvalue is
    above a critical size, then we cannot reject that there is a unit root.

    The p-values are obtained through regression surface approximation from
    MacKinnon 1994, but using the updated 2010 tables. If the p-value is close
    to significant, then the critical values should be used to judge whether
    to reject the null.

    The autolag option and maxlag for it are described in Greene.

    See the notebook `Stationarity and detrending (ADF/KPSS)
    <../examples/notebooks/generated/stationarity_detrending_adf_kpss.html>`__
    for an overview.

    References
    ----------
    .. [1] W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.

    .. [2] Hamilton, J.D.  "Time Series Analysis".  Princeton, 1994.

    .. [3] MacKinnon, J.G. 1994.  "Approximate asymptotic distribution functions for
        unit-root and cointegration tests.  `Journal of Business and Economic
        Statistics` 12, 167-76.

    .. [4] MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."  Queen"s
        University, Dept of Economics, Working Papers.  Available at
        http://ideas.repec.org/p/qed/wpaper/1227.html
    """
    x = array_like(x, "x")
    maxlag = int_like(maxlag, "maxlag", optional=True)
    regression = string_like(
        regression, "regression", options=("c", "ct", "ctt", "n")
    )
    autolag = string_like(
        autolag, "autolag", optional=True, options=("aic", "bic", "t-stat")
    )
    store = bool_like(store, "store")
    regresults = bool_like(regresults, "regresults")

    if x.max() == x.min():
        raise ValueError("Invalid input, x is constant")

    if regresults:
        store = True

    trenddict = {None: "n", 0: "c", 1: "ct", 2: "ctt"}
    if regression is None or isinstance(regression, int):
        regression = trenddict[regression]
    regression = regression.lower()
    nobs = x.shape[0]

    ntrend = len(regression) if regression != "n" else 0
    if maxlag is None:
        # from Greene referencing Schwert 1989
        maxlag = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
        # -1 for the diff
        maxlag = min(nobs // 2 - ntrend - 1, maxlag)
        if maxlag < 0:
            raise ValueError(
                "sample size is too short to use selected "
                "regression component"
            )
    elif maxlag > nobs // 2 - ntrend - 1:
        raise ValueError(
            "maxlag must be less than (nobs/2 - 1 - ntrend) "
            "where n trend is the number of included "
            "deterministic regressors"
        )
    xdiff = np.diff(x)
    xdall = lagmat(xdiff[:, None], maxlag, trim="both", original="in")
    nobs = xdall.shape[0]

    xdall[:, 0] = x[-nobs - 1 : -1]  # replace 0 xdiff with level of x
    xdshort = xdiff[-nobs:]

    if store:
        from statsmodels.stats.diagnostic import ResultsStore

        resstore = ResultsStore()
    if autolag:
        if regression != "n":
            fullRHS = add_trend(xdall, regression, prepend=True)
        else:
            fullRHS = xdall
        startlag = fullRHS.shape[1] - xdall.shape[1] + 1
        # 1 for level
        # search for lag length with smallest information criteria
        # Note: use the same number of observations to have comparable IC
        # aic and bic: smaller is better

        if not regresults:
            icbest, bestlag = _autolag(
                OLS, xdshort, fullRHS, startlag, maxlag, autolag
            )
        else:
            icbest, bestlag, alres = _autolag(
                OLS,
                xdshort,
                fullRHS,
                startlag,
                maxlag,
                autolag,
                regresults=regresults,
            )
            resstore.autolag_results = alres

        bestlag -= startlag  # convert to lag not column index

        # rerun ols with best autolag
        xdall = lagmat(xdiff[:, None], bestlag, trim="both", original="in")
        nobs = xdall.shape[0]
        xdall[:, 0] = x[-nobs - 1 : -1]  # replace 0 xdiff with level of x
        xdshort = xdiff[-nobs:]
        usedlag = bestlag
    else:
        usedlag = maxlag
        icbest = None
    if regression != "n":
        resols = OLS(
            xdshort, add_trend(xdall[:, : usedlag + 1], regression)
        ).fit()
    else:
        resols = OLS(xdshort, xdall[:, : usedlag + 1]).fit()

    adfstat = resols.tvalues[0]
    #    adfstat = (resols.params[0]-1.0)/resols.bse[0]
    # the "asymptotically correct" z statistic is obtained as
    # nobs/(1-np.sum(resols.params[1:-(trendorder+1)])) (resols.params[0] - 1)
    # I think this is the statistic that is used for series that are integrated
    # for orders higher than I(1), ie., not ADF but cointegration tests.

    # Get approx p-value and critical values
    pvalue = mackinnonp(adfstat, regression=regression, N=1)
    critvalues = mackinnoncrit(N=1, regression=regression, nobs=nobs)
    critvalues = {
        "1%": critvalues[0],
        "5%": critvalues[1],
        "10%": critvalues[2],
    }
    if store:
        resstore.resols = resols
        resstore.maxlag = maxlag
        resstore.usedlag = usedlag
        resstore.adfstat = adfstat
        resstore.critvalues = critvalues
        resstore.nobs = nobs
        resstore.H0 = (
            "The coefficient on the lagged level equals 1 - " "unit root"
        )
        resstore.HA = "The coefficient on the lagged level < 1 - stationary"
        resstore.icbest = icbest
        resstore._str = "Augmented Dickey-Fuller Test Results"
        return adfstat, pvalue, critvalues, resstore
    else:
        if not autolag:
            return adfstat, pvalue, usedlag, nobs, critvalues
        else:
            return adfstat, pvalue, usedlag, nobs, critvalues, icbest

sys.path.append('/kaggle/working/mysitepackages/')

warnings.filterwarnings('ignore') 

class ZivotAndrewsUnitRoot:
    """
    Class wrapper for Zivot-Andrews structural-break unit-root test
    """

    def __init__(self):
        """
        Critical values for the three different models specified for the
        Zivot-Andrews unit-root test.

        Notes
        -----
        The p-values are generated through Monte Carlo simulation using
        100,000 replications and 2000 data points.
        """
        self._za_critical_values = {}
        # constant-only model
        self._c = (
            (0.001, -6.78442),
            (0.100, -5.83192),
            (0.200, -5.68139),
            (0.300, -5.58461),
            (0.400, -5.51308),
            (0.500, -5.45043),
            (0.600, -5.39924),
            (0.700, -5.36023),
            (0.800, -5.33219),
            (0.900, -5.30294),
            (1.000, -5.27644),
            (2.500, -5.03340),
            (5.000, -4.81067),
            (7.500, -4.67636),
            (10.000, -4.56618),
            (12.500, -4.48130),
            (15.000, -4.40507),
            (17.500, -4.33947),
            (20.000, -4.28155),
            (22.500, -4.22683),
            (25.000, -4.17830),
            (27.500, -4.13101),
            (30.000, -4.08586),
            (32.500, -4.04455),
            (35.000, -4.00380),
            (37.500, -3.96144),
            (40.000, -3.92078),
            (42.500, -3.88178),
            (45.000, -3.84503),
            (47.500, -3.80549),
            (50.000, -3.77031),
            (52.500, -3.73209),
            (55.000, -3.69600),
            (57.500, -3.65985),
            (60.000, -3.62126),
            (65.000, -3.54580),
            (70.000, -3.46848),
            (75.000, -3.38533),
            (80.000, -3.29112),
            (85.000, -3.17832),
            (90.000, -3.04165),
            (92.500, -2.95146),
            (95.000, -2.83179),
            (96.000, -2.76465),
            (97.000, -2.68624),
            (98.000, -2.57884),
            (99.000, -2.40044),
            (99.900, -1.88932),
        )
        self._za_critical_values["c"] = np.asarray(self._c)
        # trend-only model
        self._t = (
            (0.001, -83.9094),
            (0.100, -13.8837),
            (0.200, -9.13205),
            (0.300, -6.32564),
            (0.400, -5.60803),
            (0.500, -5.38794),
            (0.600, -5.26585),
            (0.700, -5.18734),
            (0.800, -5.12756),
            (0.900, -5.07984),
            (1.000, -5.03421),
            (2.500, -4.65634),
            (5.000, -4.40580),
            (7.500, -4.25214),
            (10.000, -4.13678),
            (12.500, -4.03765),
            (15.000, -3.95185),
            (17.500, -3.87945),
            (20.000, -3.81295),
            (22.500, -3.75273),
            (25.000, -3.69836),
            (27.500, -3.64785),
            (30.000, -3.59819),
            (32.500, -3.55146),
            (35.000, -3.50522),
            (37.500, -3.45987),
            (40.000, -3.41672),
            (42.500, -3.37465),
            (45.000, -3.33394),
            (47.500, -3.29393),
            (50.000, -3.25316),
            (52.500, -3.21244),
            (55.000, -3.17124),
            (57.500, -3.13211),
            (60.000, -3.09204),
            (65.000, -3.01135),
            (70.000, -2.92897),
            (75.000, -2.83614),
            (80.000, -2.73893),
            (85.000, -2.62840),
            (90.000, -2.49611),
            (92.500, -2.41337),
            (95.000, -2.30820),
            (96.000, -2.25797),
            (97.000, -2.19648),
            (98.000, -2.11320),
            (99.000, -1.99138),
            (99.900, -1.67466),
        )
        self._za_critical_values["t"] = np.asarray(self._t)
        # constant + trend model
        self._ct = (
            (0.001, -38.17800),
            (0.100, -6.43107),
            (0.200, -6.07279),
            (0.300, -5.95496),
            (0.400, -5.86254),
            (0.500, -5.77081),
            (0.600, -5.72541),
            (0.700, -5.68406),
            (0.800, -5.65163),
            (0.900, -5.60419),
            (1.000, -5.57556),
            (2.500, -5.29704),
            (5.000, -5.07332),
            (7.500, -4.93003),
            (10.000, -4.82668),
            (12.500, -4.73711),
            (15.000, -4.66020),
            (17.500, -4.58970),
            (20.000, -4.52855),
            (22.500, -4.47100),
            (25.000, -4.42011),
            (27.500, -4.37387),
            (30.000, -4.32705),
            (32.500, -4.28126),
            (35.000, -4.23793),
            (37.500, -4.19822),
            (40.000, -4.15800),
            (42.500, -4.11946),
            (45.000, -4.08064),
            (47.500, -4.04286),
            (50.000, -4.00489),
            (52.500, -3.96837),
            (55.000, -3.93200),
            (57.500, -3.89496),
            (60.000, -3.85577),
            (65.000, -3.77795),
            (70.000, -3.69794),
            (75.000, -3.61852),
            (80.000, -3.52485),
            (85.000, -3.41665),
            (90.000, -3.28527),
            (92.500, -3.19724),
            (95.000, -3.08769),
            (96.000, -3.03088),
            (97.000, -2.96091),
            (98.000, -2.85581),
            (99.000, -2.71015),
            (99.900, -2.28767),
        )
        self._za_critical_values["ct"] = np.asarray(self._ct)

    def _za_crit(self, stat, model="c"):
        """
        Linear interpolation for Zivot-Andrews p-values and critical values

        Parameters
        ----------
        stat : float
            The ZA test statistic
        model : {"c","t","ct"}
            The model used when computing the ZA statistic. "c" is default.

        Returns
        -------
        pvalue : float
            The interpolated p-value
        cvdict : dict
            Critical values for the test statistic at the 1%, 5%, and 10%
            levels

        Notes
        -----
        The p-values are linear interpolated from the quantiles of the
        simulated ZA test statistic distribution
        """
        table = self._za_critical_values[model]
        pcnts = table[:, 0]
        stats = table[:, 1]
        # ZA cv table contains quantiles multiplied by 100
        pvalue = np.interp(stat, stats, pcnts) / 100.0
        cv = [1.0, 5.0, 10.0]
        crit_value = np.interp(cv, pcnts, stats)
        cvdict = {
            "1%": crit_value[0],
            "5%": crit_value[1],
            "10%": crit_value[2],
        }
        return pvalue, cvdict

    def _quick_ols(self, endog, exog):
        """
        Minimal implementation of LS estimator for internal use
        """
        xpxi = np.linalg.inv(exog.T.dot(exog))
        xpy = exog.T.dot(endog)
        nobs, k_exog = exog.shape
        b = xpxi.dot(xpy)
        e = endog - exog.dot(b)
        sigma2 = e.T.dot(e) / (nobs - k_exog)
        return b / np.sqrt(np.diag(sigma2 * xpxi))

    def _format_regression_data(self, series, nobs, const, trend, cols, lags):
        """
        Create the endog/exog data for the auxiliary regressions
        from the original (standardized) series under test.
        """
        # first-diff y and standardize for numerical stability
        endog = np.diff(series, axis=0)
        endog /= np.sqrt(endog.T.dot(endog))
        series /= np.sqrt(series.T.dot(series))
        # reserve exog space
        exog = np.zeros((endog[lags:].shape[0], cols + lags))
        exog[:, 0] = const
        # lagged y and dy
        exog[:, cols - 1] = series[lags : (nobs - 1)]
        exog[:, cols:] = lagmat(endog, lags, trim="none")[
            lags : exog.shape[0] + lags
        ]
        return endog, exog

    def _update_regression_exog(
        self, exog, regression, period, nobs, const, trend, cols, lags
    ):
        """
        Update the exog array for the next regression.
        """
        cutoff = period - (lags + 1)
        if regression != "t":
            exog[:cutoff, 1] = 0
            exog[cutoff:, 1] = const
            exog[:, 2] = trend[(lags + 2) : (nobs + 1)]
            if regression == "ct":
                exog[:cutoff, 3] = 0
                exog[cutoff:, 3] = trend[1 : (nobs - period + 1)]
        else:
            exog[:, 1] = trend[(lags + 2) : (nobs + 1)]
            exog[: (cutoff - 1), 2] = 0
            exog[(cutoff - 1) :, 2] = trend[0 : (nobs - period + 1)]
        return exog

    def run(self, x, trim=0.15, maxlag=None, regression="c", autolag="AIC"):
        """
        Zivot-Andrews structural-break unit-root test.

        The Zivot-Andrews test tests for a unit root in a univariate process
        in the presence of serial correlation and a single structural break.

        Parameters
        ----------
        x : array_like
            The data series to test.
        trim : float
            The percentage of series at begin/end to exclude from break-period
            calculation in range [0, 0.333] (default=0.15).
        maxlag : int
            The maximum lag which is included in test, default is
            12*(nobs/100)^{1/4} (Schwert, 1989).
        regression : {"c","t","ct"}
            Constant and trend order to include in regression.

            * "c" : constant only (default).
            * "t" : trend only.
            * "ct" : constant and trend.
        autolag : {"AIC", "BIC", "t-stat", None}
            The method to select the lag length when using automatic selection.

            * if None, then maxlag lags are used,
            * if "AIC" (default) or "BIC", then the number of lags is chosen
              to minimize the corresponding information criterion,
            * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
              lag until the t-statistic on the last lag length is significant
              using a 5%-sized test.

        Returns
        -------
        zastat : float
            The test statistic.
        pvalue : float
            The pvalue based on MC-derived critical values.
        cvdict : dict
            The critical values for the test statistic at the 1%, 5%, and 10%
            levels.
        baselag : int
            The number of lags used for period regressions.
        bpidx : int
            The index of x corresponding to endogenously calculated break period
            with values in the range [0..nobs-1].

        Notes
        -----
        H0 = unit root with a single structural break

        Algorithm follows Baum (2004/2015) approximation to original
        Zivot-Andrews method. Rather than performing an autolag regression at
        each candidate break period (as per the original paper), a single
        autolag regression is run up-front on the base model (constant + trend
        with no dummies) to determine the best lag length. This lag length is
        then used for all subsequent break-period regressions. This results in
        significant run time reduction but also slightly more pessimistic test
        statistics than the original Zivot-Andrews method, although no attempt
        has been made to characterize the size/power trade-off.

        References
        ----------
        .. [1] Baum, C.F. (2004). ZANDREWS: Stata module to calculate
           Zivot-Andrews unit root test in presence of structural break,"
           Statistical Software Components S437301, Boston College Department
           of Economics, revised 2015.

        .. [2] Schwert, G.W. (1989). Tests for unit roots: A Monte Carlo
           investigation. Journal of Business & Economic Statistics, 7:
           147-159.

        .. [3] Zivot, E., and Andrews, D.W.K. (1992). Further evidence on the
           great crash, the oil-price shock, and the unit-root hypothesis.
           Journal of Business & Economic Studies, 10: 251-270.
        """
        x = array_like(x, "x")
        trim = float_like(trim, "trim")
        maxlag = int_like(maxlag, "maxlag", optional=True)
        regression = string_like(
            regression, "regression", options=("c", "t", "ct")
        )
        autolag = string_like(
            autolag, "autolag", options=("aic", "bic", "t-stat"), optional=True
        )
        if trim < 0 or trim > (1.0 / 3.0):
            raise ValueError("trim value must be a float in range [0, 1/3)")
        nobs = x.shape[0]
        if autolag:
            adf_res = adfuller(
                x, maxlag=maxlag, regression="ct", autolag=autolag
            )
            baselags = adf_res[2]
        elif maxlag:
            baselags = maxlag
        else:
            baselags = int(12.0 * np.power(nobs / 100.0, 1 / 4.0))
        trimcnt = int(nobs * trim)
        start_period = trimcnt
        end_period = nobs - trimcnt
        if regression == "ct":
            basecols = 5
        else:
            basecols = 4
        # normalize constant and trend terms for stability
        c_const = 1 / np.sqrt(nobs)
        t_const = np.arange(1.0, nobs + 2)
        t_const *= np.sqrt(3) / nobs ** (3 / 2)
        # format the auxiliary regression data
        endog, exog = self._format_regression_data(
            x, nobs, c_const, t_const, basecols, baselags
        )
        # iterate through the time periods
        stats = np.full(end_period + 1, np.inf)
        for bp in range(start_period + 1, end_period + 1):
            # update intercept dummy / trend / trend dummy
            exog = self._update_regression_exog(
                exog,
                regression,
                bp,
                nobs,
                c_const,
                t_const,
                basecols,
                baselags,
            )
            # check exog rank on first iteration
            if bp == start_period + 1:
                o = OLS(endog[baselags:], exog, hasconst=1).fit()
                if o.df_model < exog.shape[1] - 1:
                    raise ValueError(
                        "ZA: auxiliary exog matrix is not full rank.\n"
                        "  cols (exc intercept) = {}  rank = {}".format(
                            exog.shape[1] - 1, o.df_model
                        )
                    )
                stats[bp] = o.tvalues[basecols - 1]
            else:
                stats[bp] = self._quick_ols(endog[baselags:], exog)[
                    basecols - 1
                ]
        # return best seen
        zastat = np.min(stats)
        bpidx = np.argmin(stats) - 1
        crit = self._za_crit(zastat, regression)
        pval = crit[0]
        cvdict = crit[1]
        return zastat, pval, cvdict, baselags, bpidx

    def __call__(
        self, x, trim=0.15, maxlag=None, regression="c", autolag="AIC"
    ):
        return self.run(
            x, trim=trim, maxlag=maxlag, regression=regression, autolag=autolag
        )


zivot_andrews = ZivotAndrewsUnitRoot()
zivot_andrews.__doc__ = zivot_andrews.run.__doc__

# URLs for the datasets
cocoa_prices_url = 'https://raw.githubusercontent.com/harrymmurphy/cocoa_pound/main/Cocoa_Historical.csv'
pound_prices_url = 'https://raw.githubusercontent.com/harrymmurphy/cocoa_pound/main/Pound_Historical.csv'

# Read data from URLs
cocoa_prices = pd.read_csv(cocoa_prices_url)
pound_prices = pd.read_csv(pound_prices_url)

# Convert 'Date' columns to datetime
cocoa_prices['Date'] = pd.to_datetime(cocoa_prices['Date'])
pound_prices['Date'] = pd.to_datetime(pound_prices['Date'])

# Set 'Date' as index for both datasets
cocoa_prices.set_index('Date', inplace=True)
pound_prices.set_index('Date', inplace=True)

# Define a function to handle conversion of comma-separated numeric strings to float
def convert_numeric_string_to_float(value):
    try:
        if isinstance(value, str):
            return float(value.replace(',', ''))
        else:
            return float(value)
    except ValueError:
        return float('NaN')  # or handle the error as appropriate for your use case

# Apply the conversion function to 'Close' columns
cocoa_prices['Close'] = cocoa_prices['Close'].apply(convert_numeric_string_to_float)
pound_prices['Close'] = pound_prices['Close'].apply(convert_numeric_string_to_float)

# Resample to weekly frequency
cocoa_prices_weekly = cocoa_prices.resample('W').mean()
pound_prices_weekly = pound_prices.resample('W').mean()


cocoa_prices_weekly['Weekly_Return_Cocoa'] = cocoa_prices_weekly['Close'].pct_change() * 100
pound_prices_weekly['Weekly_Return_Pound'] = pound_prices_weekly['Close'].pct_change() * 100

# Drop NaN values
cocoa_prices_weekly.dropna(inplace=True)
pound_prices_weekly.dropna(inplace=True)

# Extract relevant columns for correlation analysis
cocoa_changes = cocoa_prices_weekly['Weekly_Return_Cocoa']
pound_changes = pound_prices_weekly['Weekly_Return_Pound']

# Initialize variables to track maximum negative correlation and optimal lag
max_neg_corr = 0
optimal_lag = 0

# Test various lags to find optimal lag that maximizes negative correlation
for lag in range(1, 25):  # testing lags from 1 to 24 weeks
    # Align arrays for correlation calculation
    aligned_pound_changes = pound_changes.shift(-lag).dropna()
    aligned_cocoa_changes = cocoa_changes.loc[aligned_pound_changes.index]
    
    # Calculate correlation coefficient
    cross_corr = np.corrcoef(aligned_pound_changes, aligned_cocoa_changes)[0, 1]
    
    # Check if the correlation coefficient is less than the current maximum
    if cross_corr < max_neg_corr:
        max_neg_corr = cross_corr
        optimal_lag = lag

# Print the optimal lag and maximum negative correlation found
print(f"Optimal Lag: {optimal_lag} weeks")
print(f"Maximum Negative Correlation: {max_neg_corr:.4f}")

# Extract relevant columns for correlation analysis
cocoa_changes = cocoa_prices_weekly['Weekly_Return_Cocoa']
pound_changes = pound_prices_weekly['Weekly_Return_Pound']

# Initialize variables to track maximum positive correlation and optimal lag
max_pos_corr = -1.0
optimal_lag = 0

# Test various lags to find optimal lag that maximizes positive correlation
for lag in range(1, 60):  # testing lags from 1 to 24 weeks
    # Align arrays for correlation calculation
    aligned_pound_changes = pound_changes.shift(-lag).dropna()
    aligned_cocoa_changes = cocoa_changes.loc[aligned_pound_changes.index]
    
    # Calculate correlation coefficient
    cross_corr = np.corrcoef(aligned_pound_changes, aligned_cocoa_changes)[0, 1]
    
    # Check if the correlation coefficient is greater than the current maximum
    if cross_corr > max_pos_corr:
        max_pos_corr = cross_corr
        optimal_lag = lag

# Print the optimal lag and maximum positive correlation found
print(f"Optimal Lag: {optimal_lag} weeks")
print(f"Maximum Positive Correlation: {max_pos_corr:.4f}")
# Define the threshold for positive swing in pound prices (e.g., 3%)
positive_swing_threshold = 3.0

# Identify weeks with positive swings in pound prices
weeks_with_positive_swing = pound_prices_weekly[pound_prices_weekly['Weekly_Return_Pound'] > positive_swing_threshold]

# Initialize a list to store subsequent cocoa price changes
cocoa_price_changes_after_positive_swing = []

# Loop through each week with positive swing in pound prices
for index, row in weeks_with_positive_swing.iterrows():
    # Calculate subsequent week's percentage change in cocoa prices
    next_week_cocoa_change = cocoa_prices_weekly.loc[index + pd.DateOffset(weeks=1)]['Weekly_Return_Cocoa']
    cocoa_price_changes_after_positive_swing.append(next_week_cocoa_change)

# Convert list to Series for easier analysis
cocoa_price_changes_after_positive_swing = pd.Series(cocoa_price_changes_after_positive_swing)

# Print statistics on cocoa price changes after positive swings in pound prices
print("Statistics of Cocoa Price Changes after Positive Swings in Pound Prices:")
print("-----------------------------------------------------------------------")
print(f"Mean: {cocoa_price_changes_after_positive_swing.mean():.2f}%")
print(f"Median: {cocoa_price_changes_after_positive_swing.median():.2f}%")
print(f"Standard Deviation: {cocoa_price_changes_after_positive_swing.std():.2f}%")
print(f"Minimum: {cocoa_price_changes_after_positive_swing.min():.2f}%")
print(f"Maximum: {cocoa_price_changes_after_positive_swing.max():.2f}%")


# Sample data initialization (replace with your actual data)
pound_prices_weekly = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=50, freq='W'),
    'Weekly_Return_Pound': np.random.uniform(-1, 3, 50)  # Random data for demonstration
})

cocoa_prices_weekly = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=50, freq='W'),
    'Weekly_Return_Cocoa': np.random.uniform(-1, 3, 50)  # Random data for demonstration
})

# Parameters
positive_swing_thresholds = [1.0, 2.0, 3.0, 4.0]  # Define thresholds for positive swings in percent
lag_periods = [1, 2, 4, 8, 20, 30, 40]  # Define lag periods in weeks

# Initialize a dictionary to store results
results = {}

# Loop through each threshold and lag period
for threshold in positive_swing_thresholds:
    for lag in lag_periods:
        # Identify weeks with positive swings in pound prices based on the threshold
        weeks_with_positive_swing = pound_prices_weekly[pound_prices_weekly['Weekly_Return_Pound'] > threshold]

        # Initialize a list to store subsequent cocoa price changes
        cocoa_price_changes = []

        # Loop through each week with positive swing in pound prices
        for index, row in weeks_with_positive_swing.iterrows():
            # Calculate subsequent week's percentage change in cocoa prices with lag
            next_week_index = index + lag
            if next_week_index < len(cocoa_prices_weekly):
                next_week_cocoa_change = cocoa_prices_weekly.loc[next_week_index, 'Weekly_Return_Cocoa']
                cocoa_price_changes.append(next_week_cocoa_change)

        # Convert list to Series for easier analysis
        cocoa_price_changes = pd.Series(cocoa_price_changes)

        # Store results in the dictionary
        results[f"{threshold}% Swing, {lag}-Week Lag"] = {
            'Mean': cocoa_price_changes.mean(),
            'Median': cocoa_price_changes.median(),
            'Standard Deviation': cocoa_price_changes.std(),
            'Minimum': cocoa_price_changes.min(),
            'Maximum': cocoa_price_changes.max()
        }

# Print statistics for each combination of threshold and lag
print("Statistics of Cocoa Price Changes after Positive Swings in Pound Prices:")
print("-----------------------------------------------------------------------")
for key, stats in results.items():
    print(key)
    print(f"Mean: {stats['Mean']:.2f}%")
    print(f"Median: {stats['Median']:.2f}%")
    print(f"Standard Deviation: {stats['Standard Deviation']:.2f}%")
    print(f"Minimum: {stats['Minimum']:.2f}%")
    print(f"Maximum: {stats['Maximum']:.2f}%")
    print("---------------------------------------------")
    
merged_data = pd.merge(pound_prices_weekly, cocoa_prices_weekly, on='Date', how='inner')

# Log normalize the price columns
merged_data['Log_Return_Pound'] = np.log(merged_data['Weekly_Return_Pound'])
merged_data['Log_Return_Cocoa'] = np.log(merged_data['Weekly_Return_Cocoa'])

# Create the log_merged_data dataframe
log_merged_data = merged_data[['Date', 'Log_Return_Pound', 'Log_Return_Cocoa']]

log_merged_data = merged_data.dropna()

log_merged_data = log_merged_data[['Date', 'Log_Return_Pound', 'Log_Return_Cocoa']]

# Display the first few rows of log_merged_data
print(log_merged_data.head())


### VAR and Granger from previous code

# Merge data on 'Date'
merged_data = pd.merge(pound_prices_weekly, cocoa_prices_weekly, on='Date', how='inner')

# Log normalize the price columns
merged_data['Log_Return_Pound'] = np.log(merged_data['Weekly_Return_Pound'] + 1)  # Adding 1 to avoid log(0)
merged_data['Log_Return_Cocoa'] = np.log(merged_data['Weekly_Return_Cocoa'] + 1)  # Adding 1 to avoid log(0)

# Drop NaN values after log normalization
log_merged_data = merged_data.dropna()

# Display the first few rows of log_merged_data
print(log_merged_data.head())



# Apply the conversion function to 'Close' columns
cocoa_prices['Close'] = cocoa_prices['Close'].apply(convert_numeric_string_to_float)
pound_prices['Close'] = pound_prices['Close'].apply(convert_numeric_string_to_float)

# Resample to weekly frequency
cocoa_prices_weekly = cocoa_prices.resample('W').mean()
pound_prices_weekly = pound_prices.resample('W').mean()

# Calculate weekly percentage change for cocoa and pound prices
cocoa_prices_weekly['Weekly_Return_Cocoa'] = cocoa_prices_weekly['Close'].pct_change() * 100
pound_prices_weekly['Weekly_Return_Pound'] = pound_prices_weekly['Close'].pct_change() * 100

# Identify weeks with pound price shifts > 3%
pound_weeks_shift_gt_5 = pound_prices_weekly[abs(pound_prices_weekly['Weekly_Return_Pound']) > 3]

# Merge cocoa and pound prices based on week
merged_df = pd.merge(cocoa_prices_weekly, pound_weeks_shift_gt_5[['Weekly_Return_Pound']], left_index=True, right_index=True, suffixes=('_cocoa', '_pound'))

# Analyze cocoa price changes during weeks with pound shifts > 5%
print("Cocoa price changes during weeks with pound shifts > 3%:")
print(merged_df[['Close', 'Weekly_Return_Pound']])

# Example: Statistical analysis or visualization can be done further
# For example, you can calculate average cocoa price change during these weeks
avg_cocoa_change = merged_df['Weekly_Return_Cocoa'].mean()
print(f"\nAverage cocoa price change during these weeks: {avg_cocoa_change:.2f}%")




# Initialize VAR model
model = VAR(log_merged_data[['Log_Return_Pound', 'Log_Return_Cocoa']])
results = model.fit(maxlags=7, ic='aic')  # Adjust maxlags as needed

# Display VAR model summary
print(results.summary())

# Plot residuals to check for autocorrelation and heteroskedasticity
results.plot()

# Conduct residual diagnostics
results.plot_acorr()

# Example of differencing to achieve stationarity (if needed)
log_merged_data_diff = log_merged_data.diff().dropna()

# Fit VAR model on differenced data
model_diff = VAR(log_merged_data_diff[['Log_Return_Pound', 'Log_Return_Cocoa']])
results_diff = model_diff.fit(maxlags=7, ic='aic')

# Display VAR model summary for differenced data
print(results_diff.summary())

# Plot residuals for differenced data
results_diff.plot()

# Conduct Granger causality test on original or differenced data
maxlag = 15

def grangers_causation_matrix(data, variables, maxlag=maxlag, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
            if verbose: 
                print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

# Example of Granger causality test on original data
granger_result = grangers_causation_matrix(log_merged_data, variables=log_merged_data.columns)
print("Granger Causality Test Results:")
print(granger_result)

# Example of Granger causality test on differenced data
granger_result_diff = grangers_causation_matrix(log_merged_data_diff, variables=log_merged_data_diff.columns)
print("Granger Causality Test Results (Differenced Data):")
print(granger_result_diff)

cocoa_prices_weekly['Weekly_Return_Cocoa'] = cocoa_prices_weekly['Close'].pct_change() * 100
pound_prices_weekly['Weekly_Return_Pound'] = pound_prices_weekly['Close'].pct_change() * 100

## Zivot-Andrews
ZA_1 = ZivotAndrewsUnitRoot.run(log_merged_data)
print(ZA_1)
ZA_2 = ZivotAndrewsUnitRoot.run(log_merged_data)
print(ZA_2)
ZA_3 = ZivotAndrewsUnitRoot.run(log_merged_data)
print(ZA_3)
