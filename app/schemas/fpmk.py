from typing import Optional

from pydantic import BaseModel, validator
from datetime import date, datetime


# Shared properties
class FpmkBase(BaseModel):
    system: str
    subsystem: Optional[str] = None
    fpmk: int
    date: date


class FpmkCreate(FpmkBase):
    system: str
    subsystem: str
    fpmk: int
    date: date


class Fpmk(FpmkBase):
    pass
