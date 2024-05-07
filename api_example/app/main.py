
from dill import load
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

with open('/code/app/rfg_model.pkl', 'rb') as f:
    reloaded_model = load(f)

app = FastAPI()

class Payload(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

@app.post("/")
def predict(payload: Payload):
    df = pd.DataFrame([payload.model_dump().values()], columns=payload.model_dump().keys())
    print(df)
    y_pred = reloaded_model.predict(df)
    response = {
        'prediction': y_pred[0],
        'model_name': 'rfg_model_v1',
        'model_last_updated': '2024_05_07',
    }
    return response
