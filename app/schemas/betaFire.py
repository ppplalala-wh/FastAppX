from typing import Optional

from pydantic import BaseModel, validator
from datetime import date, datetime


# Shared properties
class BetaFireBase(BaseModel):
    alpha: float
    beta: float
    k: float
    logc: Optional[float] = 0.
    logm: Optional[float] = 0.


class BetaFireCreate(BetaFireBase):
    alpha: float
    beta: float
    k: float
    logc: Optional[float] = 0.
    logm: Optional[float] = 0.


class BetaFire(BetaFireBase):
    pass
