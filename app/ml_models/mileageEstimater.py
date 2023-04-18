from app.schemas.fpmk import Fpmk, FpmkCreate
from sklearn.linear_model import LinearRegression
from fastapi.encoders import jsonable_encoder
from typing import List
import pandas as pd
import numpy as np
import json


class MileageEstimater():
    def __init__(self):
        self.fpmk_hist = None
        self.freq = None
        self.x = None
        self.fitted = False
        self.model = LinearRegression()

    def fit(self, x: List[Fpmk], freq='M') -> None:
        # convert list of fpmk records into dataframe
        assert self.fitted is not True, f"""Mileage estimater has already been fitted.."""
        fpmk_hist = pd.DataFrame(data=jsonable_encoder(x))
        fpmk_hist['date'] = pd.to_datetime(fpmk_hist['date'])
        self.model.fit(X=fpmk_hist.index.values.reshape(-1, 1), y=fpmk_hist['mileage'].values)
        self.fpmk_hist = fpmk_hist.set_index('date').asfreq(freq).copy()
        self.freq = freq
        self.fitted = True

    def predict(self, num_cycles: int) -> List[FpmkCreate]:
        assert self.fitted is True, f"""Mileage estimater has NOT been fitted.."""
        fcst_start = self.fpmk_hist.index.shift(1)[-1]
        fcst_end = self.fpmk_hist.index.shift(num_cycles)[-1]
        x0 = len(self.fpmk_hist)
        xn = len(self.fpmk_hist) + num_cycles
        fcst_mileage = self.model.predict(np.arange(x0, xn).reshape(-1, 1))
        mileage_index = pd.date_range(start=fcst_start, end=fcst_end, freq=self.freq)
        res = pd.DataFrame(data=fcst_mileage, index=mileage_index, columns=['mileage'])
        res = res.reset_index().rename(columns={'index': 'date'})
        for col in self.fpmk_hist.columns:
            if col not in res.columns:
                res[col] = self.fpmk_hist[col].mode().values[0]
        res['fpmk'] = np.NAN

        res_list = []
        for item in json.loads(res.to_json(orient='records')):
            res_list.append(FpmkCreate(**item))
        return res_list