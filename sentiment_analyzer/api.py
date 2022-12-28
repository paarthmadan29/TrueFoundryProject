from typing import Dict

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel

from .classifier.model import Model, get_model

app = FastAPI()


class SentimentResponse(BaseModel):
    probabilities: Dict[str, float]
    sentiment: str
    confidence: float


@app.get("/")
def read_root():
    return {"TrueFoundry": "Internship Project"}


@app.post("/predict", response_model=SentimentResponse)
def predict(review:str = Query(None, title="Airline Review", decription="Enter the review for airline"), model: Model = Depends(get_model)):
    if type(review) != str:
        raise HTTPException(status_code=404, detail="Bad request")
    sentiment, confidence, probabilities = model.predict(review)
    return SentimentResponse(
        sentiment=sentiment, confidence=confidence, probabilities=probabilities
    )



