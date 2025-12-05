from pydantic import BaseModel
from datetime import datetime
from pydantic import Field
from typing import Dict 

class PredictionResponse(BaseModel):
    """Response model for fraud prediction endpoint"""
    prediction: str = Field(
        ..., 
        description="Fraud prediction: 'fraud' or 'not_fraud'"
    )
    proba: Dict[str, float] = Field(
        ...,
        description="Probability scores for each class"
    )
    
    #Documentation Example ffor Swagger UI
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "not_fraud",
                "proba": {
                    "not_fraud": 0.92,
                    "fraud": 0.08
                },

            }
        }

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True
            }
        }



class InputData(BaseModel):
    """
    InputData represents the structure of a transaction record.

    Attributes:
        merchant (str): Name of the merchant
        category (str): Category of the transaction
        city (str): City where transaction occurred
        state (str): State where transaction occurred
        job (str): Job of the customer
        amt (int): Transaction amount
        lat (float): Latitude of transaction location
        long (float): Longitude of transaction location
        city_pop (int): Population of the city
        trans_date_trans_time (datetime): Date and time of transaction

    Example:
        {
            "merchant": "Walmart",
            "category": "groceries",
            "city": "Springfield",
            "state": "IL",
            "job": "Engineer",
            "amt": 120,
            "lat": 39.7817,
            "long": -89.6501,
            "city_pop": 116250,
            "trans_date_trans_time": "2025-12-04T14:30:00"
        }
    """
    merchant: str
    category: str
    city: str
    state: str
    job: str
    amt: float
    lat: float
    long: float
    city_pop: int
    trans_date_trans_time: datetime


class ProcessedInputData(BaseModel):
    """
    ProcessedInputData represents the processed transaction record with datetime fields broken down.

    Attributes:
        merchant (str): Name of the merchant
        category (str): Category of the transaction
        city (str): City where transaction occurred
        state (str): State where transaction occurred
        job (str): Job of the customer
        amt (int): Transaction amount
        lat (float): Latitude of transaction location
        long (float): Longitude of transaction location
        city_pop (int): Population of the city
        trans_hour (int): Hour of the transaction (0-23)
        trans_day (int): Day of the month (1-31)
        trans_month (int): Month of the transaction (1-12)
        trans_weekday (int): Day of the week (0=Monday, 6=Sunday)
    """
    merchant: str
    category: str
    city: str
    state: str
    job: str
    amt: float
    lat: float
    long: float
    city_pop: int
    trans_hour: int
    trans_day: int
    trans_month: int
    trans_weekday: int