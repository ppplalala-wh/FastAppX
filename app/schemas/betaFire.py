from typing import Optional

from pydantic import BaseModel, validator
from datetime import date, datetime


# Shared properties
class BetaFireBase(BaseModel):
    a: float
    b: float
    k: float
    logc: Optional[float] = None
    logm: Optional[float] = None


class BetaFireCreate(BetaFireBase):
    a: float
    b: float
    k: float
    logc: Optional[float] = None
    logm: Optional[float] = None


class BetaFire(BetaFireBase):
    pass
