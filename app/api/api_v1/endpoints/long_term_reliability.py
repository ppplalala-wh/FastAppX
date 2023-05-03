from typing import Any, List
from fastapi import APIRouter
from app.ml_models.betaFIRE import BetaFire, defaultInit, defaultBounds
from app.ml_models.mileageEstimater import MileageEstimater
import numpy as np
import json

from app import schemas

router = APIRouter()


@router.post("/train", response_model=None)
def train_model(
        *,
        fpmks_in: List[schemas.FpmkCreate],
        design_oem_mileage: int
) -> Any:
    # fit betaFire model
    betaFiremodel = BetaFire()
    model, id = betaFiremodel.fit(fpmks_in, init=defaultInit.params, bounds=defaultBounds.bounds,
                                  fixed={'logc': np.log(design_oem_mileage)})
    return {'best_params': model, 'model_id': id}


@router.post("/forecast_by_mileage", response_model=None)
def forecast_by_mileage(
        *,
        model_id: str,
        mileage: int
) -> Any:
    # fit betaFire model
    betaFiremodel = BetaFire()
    if betaFiremodel.load(mod_id=model_id):
        fpmk = betaFiremodel.predict_by_mileage(mileage=mileage)
        return {'fpmk': fpmk}
    else:
        return 'model not found'


@router.post("/forecast_by_cycles", response_model=List[schemas.FpmkCreate])
def forecast_by_cycles(
        *,
        mod_id: str,
        no_fcst_cycles: int
) -> Any:
    # fit betaFire model
    betaFiremodel = BetaFire()
    betaFiremodel.load(mod_id)
    # forecast mileage,
    mileage_estimater = MileageEstimater()
    fpmks_in = []
    for item in json.loads(betaFiremodel.data.to_json(orient='records')):
        fpmks_in.append(schemas.FpmkCreate(**item))
    mileage_estimater.fit(fpmks_in)
    # range of forecast
    fcst_range = mileage_estimater.predict(num_cycles=no_fcst_cycles)
    # fcst reliability
    res = betaFiremodel.predict(fcst_range)
    return res


@router.post("/forecast", response_model=List[schemas.FpmkCreate])
def forecast(
        *,
        mod_id: str,
        input_fpmk: List[schemas.FpmkCreate]
) -> Any:
    betaFiremodel = BetaFire()
    betaFiremodel.load(mod_id)
    res = betaFiremodel.predict(input_fpmk)
    return res
