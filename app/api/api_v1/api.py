from fastapi import APIRouter

from app.api.api_v1.endpoints import long_term_reliability, feature_engineering, data_sourcing

api_router = APIRouter()
api_router.include_router(data_sourcing.router, prefix="/data_sourcing", tags=["data_sourcing"])
api_router.include_router(feature_engineering.router, prefix="/transformer", tags=["feature_engineering"])
api_router.include_router(long_term_reliability.router, prefix="/lt_reliability", tags=["reliability"])