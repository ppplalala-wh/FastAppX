from abc import ABC
import numpy as np
from typing import List, Union, Tuple
import pandas as pd

from app.schemas import FpmkCreate
from app.schemas.fpmk import FpmkCreate
from app.schemas.betaFire import BetaFireCreate
from scipy.optimize import minimize, Bounds
from scipy.special import beta, betainc
from functools import partial
from fastapi.encoders import jsonable_encoder
import datetime
import json


class ParameterBounds():
    def __init__(self, dict_b: dict):
        self._bounds = dict_b

    @property
    def bounds(self):
        return self._bounds


class ParameterInit():
    def __init__(self, dict_i: dict):
        self._params = dict_i

    @property
    def params(self):
        return self._params


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

    @staticmethod
    def number_of_failures(params, x):
        a, b, k, logc, logm = params
        c = np.exp(logc)
        m = np.exp(logm)
        xprime = x * (1 - k)
        incomplete_beta_portion = betainc(a, b, k + xprime / c) * beta(a, b)
        return c * m * incomplete_beta_portion / (1 - k)

    @staticmethod
    def failure_rate(params, x):
        a, b, k, logc, logm = params
        c = np.exp(logc)
        m = np.exp(logm)
        xprime = x * (1 - k)
        res = m*np.power(k + xprime/c, a-1)*np.power(1 - k - (xprime/c), b-1)
        return res

    @staticmethod
    def failure_acc(params, x):
        a, b, k, logc, logm = params
        c = np.exp(logc)
        m = np.exp(logm)
        xprime = x * (1 - k)
        common = m / c * (1 - k) * np.power(k + xprime / c, a - 2) * np.power(1 - k - xprime / c, b - 2)
        return common * ((a - 1) * (1 - k - xprime / c) - (b - 1) * (k + xprime))

    def _check_fit(func):
        def wrap(self, *args, **kwargs):
            if not self.fitted:
                raise Exception('Model has not been successfully fitted')
            return func(self, *args, **kwargs)

        return wrap

    def fit(self, x: List[FpmkCreate], init: ParameterInit, bounds: ParameterBounds,
            method='failure-rate', verbose=0, fixed: dict = None):
        fpmk_hist = pd.DataFrame(data=jsonable_encoder(x))
        fpmk_hist['date'] = pd.to_datetime(fpmk_hist['date'])
        s = fpmk_hist[['mileage', 'fpmk']].set_index('mileage').squeeze()
        assert method in ['failure-rate', 'number-of-failure']
        # checking parameters
        for key in bounds.keys():
            assert key in self.opt_params, f"""Bounds param name must be in the list of params in Fire model: {self.opt_params} """
        for key in init.keys():
            assert key in self.opt_params, f"""Init param name must be in the list of params in Fire model: {self.opt_params} """
        assert bounds.keys() == init.keys(), f"""An initial value must be given for each parameter in input bounds"""
        if fixed:
            for key in fixed.keys():
                assert key in self.opt_params, f"""Fixed bound param name must be in the list of params in Fire model: {self.opt_params} """
                assert key not in bounds.keys(), f"""No bound should be specified for fixed parameters, param_name: {key} """
                assert key not in init.keys(), f"""No initial value should be specified for fixed parameters, param_name: {key} """
        self.fixed_params = fixed
        if s.isnull().any():
            if verbose:
                print('Removing NaN from input series...')
            s = s.dropna()

        if s.index.isnull().any():
            if verbose:
                print('Removing NaN from index...')
            s = s[s.index.notnull()]
        self.series = s
        if method == 'number-of-failure':
            func = BetaFire.number_of_failures
        else:
            func = BetaFire.failure_rate
        # make sure parameters orders are the same in both initial and bounds
        # a workaround to fix lb, ub and x0 for fixed params since way of dynamically tuning selected
        # parameters using scipy.optimize is not clear
        if fixed:
            x0 = [fixed.get(p, init.get(p)) for p in self.opt_params]
            lb = [fixed.get(p, min(bounds.get(p, [None]))) for p in self.opt_params]
            # add a small delta on upper bound to make optimization function work
            ub = [fixed.get(p, max(bounds.get(p, [None]))) + 0.001 for p in self.opt_params]
        else:
            x0 = [init.get(p, init.get(p)) for p in self.opt_params]
            lb = [min(bounds.get(p, [None])) for p in self.opt_params]
            ub = [max(bounds.get(p, [None])) for p in self.opt_params]
        out = minimize(
            partial(self.objective, f=func), x0=x0, args=(s.index, s.values), bounds=Bounds(lb, ub),
            options=self._opt_options
        )
        if not out.success:
            if not isinstance(out.message, str):
                message = out.message.decode()
            else:
                message = out.message
            print(f'Optimisation unsuccessful: {message}')
            self.fitted = True
        else:
            self.fitted = True

        for param, val in zip(self.opt_params, out.x):
            if param in init.keys():
                self.best_params[param] = val

    def predict(self, fcst_range: List[FpmkCreate]) -> tuple[list[FpmkCreate], BetaFireCreate]:
        fcst_range = pd.DataFrame(data=jsonable_encoder(fcst_range))
        fcst_range['date'] = pd.to_datetime(fcst_range['date'])
        params = {}
        for p in self.opt_params:
            if self.fixed_params:
                params[p] = self.best_params.get(p, self.fixed_params.get(p))
            else:
                params[p] = self.best_params.get(p)
        params = list(params.values())
        fpmk = 1e6 * pd.Series(
            BetaFire.failure_rate(params, x=fcst_range['mileage'].squeeze().to_numpy()), index=fcst_range.index, name='fpmk'
        )
        res = fcst_range.copy()
        res.loc[:, 'fpmk'] = fpmk
        res_list = []
        for item in json.loads(res.to_json(orient='records')):
            res_list.append(FpmkCreate(**item))
        return res_list, BetaFireCreate(**self.best_params)


# deterioration type bounds
det_type_bound = {
    'wearout': {'a': [1, 2], 'b': [0.1, 0.3], 'k': [0.695, 0.705], 'logm': [-25, 0]},
    'random': {'a': [0.6, 0.8], 'b': [0.6, 0.8], 'k': [0.495, 0.505], 'logc': [24, 25], 'logm': [-25, 0]},
    'fatigue': {'a': [24, 25], 'b': [0.7, 0.9], 'k': [0.895, 0.905], 'logm': [-25, 0]},
}

det_type_init = {
    'wearout': {'a': 0.7, 'b': 0.7, 'k': 0.5, 'logm': -19},
    'random': {'a': 0.7, 'b': 0.7, 'k': 0.5, 'logc': 19.5, 'logm': -19},
    'fatigue': {'a': 0.7, 'b': 0.7, 'k': 0.5, 'logm': -19}
}
defaultInit = ParameterInit(det_type_init['wearout'])
defaultBounds = ParameterBounds(det_type_bound['wearout'])
