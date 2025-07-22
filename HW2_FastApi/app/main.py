# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict_text

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    return predict_text(input.text)
