from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

import io
from pydantic import BaseModel
import pandas as pd
import cloudpickle
from typing import Any

ml_models = {}

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class ItemResponse(BaseModel):
    prediction: float


def load_model_pipeline():
    with open('model/ridge_pipeline.pkl', 'rb') as file:
        model_pipeline = cloudpickle.load(file)
    return model_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_pipeline = load_model_pipeline()
    ml_models["ridge"] = model_pipeline

    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan, title="Used Cars Prices Prediction API")


@app.post("/predict_item", response_model=ItemResponse)
async def predict_item(item: Item) -> Any:
    """
    Predicts one used car price, accepts JSON\n
    Returns response object containing prediction as float
    """
    df = pd.DataFrame([jsonable_encoder(item)])
    prediction = ml_models['ridge'].predict(df)[0]

    response = ItemResponse(prediction=prediction)
    return response

@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    """
    Predicts prices for multiple used cars, accepts cars data as CSV\n
    Returns CSV file with the same data as accepted one and with new 'selling_price_predicted' column containing predictions for each car
    """
    try:
        df = pd.read_csv(file.file)

        predictions = ml_models['ridge'].predict(df)
        df['selling_price_predicted'] = predictions
        df = df.drop('selling_price', axis=1)

        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=result.csv"

        return response

    except (BaseException, Exception):
        raise HTTPException(status_code=500, detail='Internal server error')
    finally:
        file.file.close()
