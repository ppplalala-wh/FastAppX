import datetime
import pandas as pd
from typing import Any, List, Union, Optional
from fastapi import APIRouter, Depends, HTTPException
from app.ml_models import betaFireModel
from fastapi.encoders import jsonable_encoder

from app import schemas

router = APIRouter()


@router.post("/forecast_reliability", response_model=Union[List[schemas.Fpmk], schemas.BetaFire])
def forecast_reliability(
        *,
        fpmks_in: List[schemas.FpmkCreate],
        no_years: int,
        model_in: Optional[schemas.BetaFire]
) -> Any:
    fpmks_df = pd.DataFrame(data=jsonable_encoder(fpmks_in))
    fpmks_df.head()
    fpmks_df.set_index('date')
    betaFireModel.fit(fpmks_df['fpmk'])
    fcst_res = betaFireModel.predict(no_of_forecast=no_years)
    fcst_res['system'] = fpmks_df['system'].mode()
    fcst_res['subsystem'] = fpmks_df['subsystem'].mode()

