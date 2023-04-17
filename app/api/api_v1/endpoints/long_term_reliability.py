import datetime
import pandas as pd
from typing import Any, List, Union, Optional
from fastapi import APIRouter, Depends, HTTPException
from app.ml_models import betaFireModel, defaultInit, defaultBounds
from fastapi.encoders import jsonable_encoder
import json

from app import schemas

router = APIRouter()


@router.post("/forecast_reliability", response_model=List[schemas.Fpmk])
def forecast_reliability(
        *,
        fpmks_in: List[schemas.FpmkCreate],
        no_years: int
) -> Any:
    fpmks_df = pd.DataFrame(data=jsonable_encoder(fpmks_in))
    fpmks_df['date'] = pd.to_datetime(fpmks_df['date'])
    fpmks_df.set_index('date', inplace=True)
    fpmks_df = fpmks_df.asfreq('M')
    betaFireModel.fit(fpmks_df['fpmk'], init=defaultInit, bounds=defaultBounds)
    fcst_res = betaFireModel.predict(no_of_forecast=no_years)
    fcst_res = pd.DataFrame(fcst_res)
    fcst_res.reset_index(inplace=True)
    fcst_res = fcst_res.set_axis(['date', 'fpmk'], axis=1)
    fcst_res['system'] = fpmks_df['system'].mode().values[0]
    fcst_res['subsystem'] = fpmks_df['subsystem'].mode().values[0]

    res = []
    for item in json.loads(fcst_res.to_json(orient='records')):
        res.append(schemas.Fpmk(**item))
    return res

