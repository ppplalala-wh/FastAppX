import datetime
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException
from app.ml_models.betaFIRE import betaFIRE

from app import schemas

router = APIRouter()

@router.put("/forecast", response_model=schemas.BetaFire)
def forecast_reliability(
        *,
        fpmks_in: List[schemas.FpmkCreate]
) -> Any:
    # if no parameter passed in, train and ind best model


