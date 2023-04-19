from typing import Any, List
from fastapi import APIRouter
from app.ml_models.betaFIRE import BetaFire, defaultInit, defaultBounds
from app.ml_models.mileageEstimater import MileageEstimater
import numpy as np

from app import schemas

router = APIRouter()


@router.post("/forecast_reliability", response_model=None)
def forecast_reliability(
        *,
        fpmks_in: List[schemas.FpmkCreate],
        no_fcst_cycles: int,
        design_oem_mileage: int
) -> Any:
    # forecast mileage,
    mileage_estimater = MileageEstimater()
    mileage_estimater.fit(fpmks_in)
    # range of forecast
    fcst_range = mileage_estimater.predict(num_cycles=no_fcst_cycles)
    # fit betaFire model
    betaFiremodel = BetaFire()
    betaFiremodel.fit(fpmks_in, init=defaultInit.params, bounds=defaultBounds.bounds,
                      fixed={'logc': np.log(design_oem_mileage)})
    # fcst reliability
    res, model = betaFiremodel.predict(fcst_range)
    return {'forecast_result': res, 'best_params': model}
