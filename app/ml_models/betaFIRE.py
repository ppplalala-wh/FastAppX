from abc import ABC
import numpy as np
from typing import List
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import MonthEnd
from scipy.optimize import minimize, Bounds
from scipy.special import beta, betainc
from functools import partial
import datetime
import json


class FailureRateModel(ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    def number_of_failures(params, x):
        pass

    @staticmethod
    def failure_rate(params, x):
        pass

    @staticmethod
    def failure_acc(params, x):
        pass


class BetaFire(FailureRateModel):
    # parameters for file model
    opt_params = ['a', 'b', 'k', 'logc', 'logm']

    def __init__(self):
        super().__init__()
        self.y_bar = None
        self.res = None
        self.arma = None
        self.X = None
        self._stabilizer = 1e2
        self._opt_options = {
            'disp': None, 'maxls': 20, 'iprint': -1,
            'gtol': 1e-7, 'eps': 1e-11, 'maxiter': 500,
            'ftol': 1e-7, 'maxcor': 10
        }
        self.opt_params = ['a', 'b', 'k', 'logc', 'logm']
        self.objective = lambda x, y, z, f: self._stabilizer * np.mean(np.power(1e6 * f(x, y) - z, 2))
        self.series = None
        self.fitted = False
        self.best_params = {}
        self.fixed_params = None

    def _check_fit(func):
        def wrap(self, *args, **kwargs):
            if not self.fitted:
                raise Exception('Model has not been successfully fitted')
            return func(self, *args, **kwargs)

        return wrap

    def fit(self, X: pd.Series):
        self.X = X.copy()
        self.arma = ARIMA(X, order=(0, 0, 3))
        self.res = self.arma.fit()

    def predict(self, no_of_forecast: int) -> pd.Series:
        fcst_start = self.X[-1].index.shift(1)[0]
        fcst_end = self.X[-1].index.shift(no_of_forecast)[-1]
        y_bar = self.res.predict(start=fcst_start, end=fcst_end)
        self.y_bar = y_bar.copy()

        return y_bar


betaFireModel = BetaFire()
