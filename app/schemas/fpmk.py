from typing import Optional

from pydantic import BaseModel, validator
from datetime import date, datetime


# Shared properties
class FpmkBase(BaseModel):
    system: str
    subsystem: Optional[str] = None
    mileage: int
    unit: str
    fpmk: Optional[int]
    date: date


class FpmkCreate(FpmkBase):
    fpmk: Optional[int]


class Fpmk(FpmkBase):
    pass
