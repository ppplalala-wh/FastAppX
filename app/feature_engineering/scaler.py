from typing import Union, List, TypeVar, Generic
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from app.schemas import FpmkCreate
from fastapi.encoders import jsonable_encoder
import pandas as pd
import json
import sys

ScalerType = TypeVar("ScalerType", bound=Union[MinMaxScaler, RobustScaler, StandardScaler])


class CustomisedScaler(Generic[ScalerType]):
    def __init__(self, scaler_name: str):
        self.scaler = getattr(sys.modules[__name__], scaler_name)()

    def transform(self, fpmk_train: List[FpmkCreate], fpmk_in: List[FpmkCreate]) -> List[FpmkCreate]:
        fpmk_hist = pd.DataFrame(data=jsonable_encoder(fpmk_train))
        fpmk_hist['fpmk'] = self.scaler.fit(fpmk_hist['fpmk'].to_numpy().reshape(-1, 1))
        fpmk_out = pd.DataFrame(data=jsonable_encoder(fpmk_in))
        fpmk_out['fpmk'] = self.scaler.transform(fpmk_out['fpmk'].to_numpy().reshape(-1, 1))
        res = []
        for item in json.loads(fpmk_out.to_json(orient='records')):
            res.append(FpmkCreate(**item))
        return res

    def inverse_transform(self, fpmk_train: List[FpmkCreate], fpmk_in: List[FpmkCreate]) -> List[FpmkCreate]:
        fpmk_hist = pd.DataFrame(data=jsonable_encoder(fpmk_train))
        fpmk_hist['fpmk'] = self.scaler.fit(fpmk_hist['fpmk'].to_numpy().reshape(-1, 1))
        fpmk_out = pd.DataFrame(data=jsonable_encoder(fpmk_in))
        fpmk_out['fpmk'] = self.scaler.inverse_transform(fpmk_out['fpmk'].to_numpy().reshape(-1, 1))
        res = []
        for item in json.loads(fpmk_out.to_json(orient='records')):
            res.append(FpmkCreate(**item))
        return res