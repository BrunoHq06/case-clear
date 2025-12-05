"""
Fraud Detection API
FastAPI server for credit card fraud detection using machine learning
"""

import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime
from basemodel import InputData, ProcessedInputData,HealthResponse,PredictionResponse
import numpy as np
import pandas as pd
from typing import Dict

# ============================================================================
# FastAPI App Setup
# ============================================================================
app = FastAPI(
    title="Fraud Detection API",
    description="REST API for detecting fraudulent credit card transactions",
    version="1.0.0"
)

# Load the trained model
try:
    model = joblib.load("./artifacts/fraud_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# ============================================================================
# Helper Functions
# ============================================================================
def transform_input(input_data: InputData) -> ProcessedInputData:
    """
    Transform InputData with datetime into ProcessedInputData with separated components.
    
    This function:
    1. Parses the datetime from ISO format
    2. Extracts hour, day, month, and weekday components
    3. Returns ProcessedInputData ready for model prediction
    
    Args:
        input_data (InputData): Raw transaction input with datetime field
        
    Returns:
        ProcessedInputData: Processed data with datetime decomposed
        
    Raises:
        ValueError: If datetime parsing fails
    """
    # Parse datetime
    if isinstance(input_data.trans_date_trans_time, str):
        try:
            trans_datetime = datetime.fromisoformat(input_data.trans_date_trans_time)
        except ValueError as e:
            raise ValueError(f"Invalid datetime format: {str(e)}")
    else:
        trans_datetime = input_data.trans_date_trans_time
    
    # Extract temporal features
    trans_hour = trans_datetime.hour
    trans_day = trans_datetime.day
    trans_month = trans_datetime.month
    trans_weekday = trans_datetime.weekday()
    
    # Create ProcessedInputData with all features
    processed_data = ProcessedInputData(
        merchant=input_data.merchant,
        category=input_data.category,
        city=input_data.city,
        state=input_data.state,
        job=input_data.job,
        amt=input_data.amt,
        lat=input_data.lat,
        long=input_data.long,
        city_pop=input_data.city_pop,
        trans_hour=trans_hour,
        trans_day=trans_day,
        trans_month=trans_month,
        trans_weekday=trans_weekday
    )
    
    return processed_data


# ============================================================================
# API Endpoints
# ============================================================================
@app.get(
    "/",
    tags=["Information"],
    summary="API Information",
    response_description="API metadata"
)
def root():
    """
    Get general information about the Fraud Detection API.
    
    Returns:
        dict: API name, version, description, and available endpoints
    """
    return {
        "name": "Fraud Detection API",
        "version": "1.0.0",
        "description": "Credit card fraud detection using machine learning",
        "documentation": "/docs",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict"
        }
    }


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Check API Health",
    response_description="API health status"
)
def health() -> HealthResponse:
    """
    Check if the API is running and the model is loaded.
    
    This endpoint is useful for monitoring and load balancing.
    Returns the current API status and model availability.
    
    Returns:
        HealthResponse: API status and model load status
        
    Example:
        GET /api/health
        
        Response:
        {
            "status": "healthy",
            "model_loaded": true
        }
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Predict Fraud",
    response_description="Fraud prediction with probabilities"
)
def predict(request: InputData) -> PredictionResponse:
    """
    Make a fraud prediction for a credit card transaction.
    
    This endpoint accepts transaction details and returns:
    - Binary classification (0 = legitimate, 1 = fraud)
    - Probability scores for both classes
    - Confidence level of the prediction
    
    The API automatically:
    - Validates input data
    - Parses datetime and extracts temporal features
    - Reorders features in the correct order expected by the model
    - Applies preprocessing pipeline
    - Generates predictions
    
    Args:
        request (InputData): Transaction data including:
            - merchant: Merchant name
            - category: Transaction category
            - city, state: Location information
            - job: Cardholder's job
            - amt: Transaction amount
            - lat, long: Geographic coordinates
            - city_pop: City population
            - trans_date_trans_time: ISO 8601 datetime string
        
    Returns:
        PredictionResponse: Prediction result with probabilities and confidence
        
    Raises:
        HTTPException 503: Model not loaded
        HTTPException 400: Invalid input or prediction error
        HTTPException 422: Validation error (missing fields, wrong types)
        
    Example:
        POST /api/predict
        
        Request:
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
        
        Response:
        {
            "prediction": 0,
            "probability": {
                "legitimate": 0.92,
                "fraud": 0.08
            }
    """
    # Verify model is loaded
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Transform input: InputData -> ProcessedInputData
        processed_data = transform_input(request)
        
        # Convert to dictionary
        features_dict = processed_data.dict()
        
        # Get feature order from model
        feature_order = model.feature_names_in_.tolist()
        
        # Create DataFrame with features in correct order
        features_df = pd.DataFrame([features_dict])
        features_df = features_df[feature_order]
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        
        # Convert prediction to string ("fraud" or "not_fraud")
        prediction_str = "fraud" if prediction == 1 else "not_fraud"
        
        return PredictionResponse(
            prediction=prediction_str,
            proba={
                "not_fraud": float(probabilities[0]),
                "fraud": float(probabilities[1])
            }
        )
        
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing feature in processed data: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid value: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing prediction: {str(e)}"
        )


# ============================================================================
# Application Entry Point
# ============================================================================
if __name__ == "__main__":
    print("Starting Fraud Detection API...")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
