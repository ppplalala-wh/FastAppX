from typing import List, Any
from fastapi import APIRouter, File, UploadFile
import pandas as pd
from app import schemas
import json
from datetime import date

router = APIRouter()


@router.post("/batch", response_model=List[schemas.FpmkCreate])
async def create_upload_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    file.file.close()
    res_list = []
    json_list = []
    for item in json.loads(df.to_json(orient='records')):
        res_list.append(schemas.FpmkCreate(**item))
        json_list.append(json.loads(schemas.FpmkCreate(**item).json()))
    with open(f"app/db/data.json", "w+") as out_file:
        out_file.seek(0)
        json.dump(json_list, out_file, indent=2)
    return res_list


@router.post("/transaction", response_model=List[schemas.FpmkCreate])
async def create_trx(
        *,
        system: str,
        subsystem: str,
        mileage: int,
        unit: str,
        fpmk: float,
        date: date
) -> Any:
    try:
        db_file = open(f"app/db/data.json")
    except FileNotFoundError:
        return False
    db_df = pd.DataFrame(json.load(db_file))
    db_df['date'] = pd.to_datetime(db_df['date'])
    input_data = {
        'system' : system,
        'subsystem' : subsystem,
        'mileage' : mileage,
        'unit' : unit,
        'fpmk' : fpmk,
        'date' : date
    }
    new_row = pd.DataFrame(input_data, index=[0])
    db_df = db_df.append(new_row, ignore_index=True)
    db_json = json.loads(db_df.to_json(orient='records', date_format='iso'))
    with open(f"app/db/data.json", "w+") as out_file:
        out_file.seek(0)
        json.dump(db_json, out_file, indent=2)

    res_list = []
    for item in json.loads(db_df.to_json(orient='records')):
        res_list.append(schemas.FpmkCreate(**item))
    return res_list

    return [schemas.FpmkCreate(**input_data)]


@router.get("/history", response_model=List[schemas.FpmkCreate])
async def query_all():
    try:
        db_file = open(f"app/db/data.json")
    except FileNotFoundError:
        return False
    db_df = pd.DataFrame(json.load(db_file))
    db_df['date'] = pd.to_datetime(db_df['date'])

    res_list = []
    for item in json.loads(db_df.to_json(orient='records')):
        res_list.append(schemas.FpmkCreate(**item))
    return res_list