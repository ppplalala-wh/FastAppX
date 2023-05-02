from typing import List, Any
from fastapi import APIRouter
from app.feature_engineering.scaler import CustomisedScaler

from app import schemas

router = APIRouter()


@router.post("/transform", response_model=List[schemas.FpmkCreate])
def fit_transform(
        *,
        transformer_name: str,
        training_data: List[schemas.FpmkCreate],
        input_data: List[schemas.FpmkCreate]
) -> Any:
    cs = CustomisedScaler(transformer_name)
    return cs.transform(fpmk_train=training_data, fpmk_in=input_data)


@router.post("/inverse_transform", response_model=List[schemas.FpmkCreate])
def inverse_transform(
        *,
        transformer_name: str,
        training_data: List[schemas.FpmkCreate],
        input_data: List[schemas.FpmkCreate]
) -> Any:
    cs = CustomisedScaler(transformer_name)
    return cs.inverse_transform(fpmk_train=training_data, fpmk_in=input_data)