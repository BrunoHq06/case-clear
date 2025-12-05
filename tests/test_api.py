"""
Simple tests for Fraud Detection API
"""

import pytest
from unittest.mock import patch
import numpy as np


class TestAPIEndpoints:
    """Tests for main API endpoints"""

    def test_health_check(self, client):
        """Test 1: Health check endpoint returns correct status"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)

    @patch('api.model')
    def test_predict_not_fraud(self, mock_model, client, valid_transaction):
        """Test 2: Predict endpoint correctly identifies legitimate transactions"""
        # Mock model configuration
        mock_model.feature_names_in_ = np.array([
            'merchant', 'category', 'city', 'state', 'job', 'amt',
            'lat', 'long', 'city_pop', 'trans_hour', 'trans_day',
            'trans_month', 'trans_weekday'
        ])
        mock_model.predict.return_value = np.array([0])  # Not fraud
        mock_model.predict_proba.return_value = np.array([[0.92, 0.08]])

        response = client.post("/api/predict", json=valid_transaction)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "not_fraud"
        assert data["proba"]["not_fraud"] > data["proba"]["fraud"]

    @patch('api.model')
    def test_predict_fraud(self, mock_model, client, valid_transaction):
        """Test 3: Predict endpoint correctly identifies fraudulent transactions"""
        # Mock model configuration
        mock_model.feature_names_in_ = np.array([
            'merchant', 'category', 'city', 'state', 'job', 'amt',
            'lat', 'long', 'city_pop', 'trans_hour', 'trans_day',
            'trans_month', 'trans_weekday'
        ])
        mock_model.predict.return_value = np.array([1])  # Fraud
        mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])

        response = client.post("/api/predict", json=valid_transaction)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "fraud"
        assert data["proba"]["fraud"] > data["proba"]["not_fraud"]
