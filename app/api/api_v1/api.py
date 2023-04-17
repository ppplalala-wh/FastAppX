from fastapi import APIRouter

from app.api.api_v1.endpoints import long_term_reliability

api_router = APIRouter()
api_router.include_router(long_term_reliability.router, prefix="/fpmk_fcst", tags=["reliability_fcst"])