import datetime
import pandas as pd
from typing import Any, List, Union, Optional
from fastapi import APIRouter, Depends, HTTPException
from app.ml_models.betaFIRE import BetaFire, defaultInit, defaultBounds
from app.ml_models.mileageEstimater import MileageEstimater
from fastapi.encoders import jsonable_encoder
import json

from app import schemas

router = APIRouter()


@router.post("/forecast_reliability", response_model=List[schemas.Fpmk])
def forecast_reliability(
        *,
        fpmks_in: List[schemas.FpmkCreate],
        no_cycles: int
) -> Any:
    # forecast mileage,
    mileage_estimater = MileageEstimater()
    mileage_estimater.fit(fpmks_in)
    # range of forecast
    fcst_range = mileage_estimater.predict(num_cycles=no_cycles)
    # fit betaFire model
    betaFiremodel = BetaFire()
    betaFiremodel.fit(fpmks_in, init=defaultInit.params, bounds=defaultBounds.bounds, fixed={'logc': 2.3})
    # fcst reliability
    res = betaFiremodel.predict(fcst_range)
    return res

