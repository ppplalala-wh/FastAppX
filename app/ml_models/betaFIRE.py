from abc import ABC
import numpy as np
import pandas as pd
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
        self._stabilizer = 1e2
        self._opt_options = {
            'disp': None, 'maxls': 20, 'iprint': -1,
            'gtol': 1e-7, 'eps': 1e-11, 'maxiter': 500,
            'ftol': 1e-7, 'maxcor': 10
        }
        self.opt_params = ['a', 'b', 'k', 'logc', 'logm']
        self.objective = lambda x, y, z, f: self._stabilizer*np.mean(np.power(1e6*f(x, y) - z, 2))
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


    @staticmethod

