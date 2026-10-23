from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
app = FastAPI(title="My First API")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_column = joblib.load("columns.pkl")

class CarFeature(BaseModel):
    features: dict
@app.get("/")
def read_root():
    return {"Hello": "welocme to the hoome page of the this gateway"}

@app.post("/predict")
def predict(payload: CarFeature):
    try:
        input_f = pd.DataFrame([payload.features])

        for col in feature_column:
            if col not in input_f.columns:
                input_f[col] = 0

        input_f = input_f[feature_column]

        input_s = scaler.transform(input_f)
        prediction = model.predict(input_s)[0]
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}



