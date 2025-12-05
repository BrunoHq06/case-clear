"""
Pytest configuration and shared fixtures for Fraud Detection API tests.

This file provides reusable test fixtures that are automatically
discovered by pytest and made available to all test files in this directory.
"""

import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient
from api import app


@pytest.fixture
def client():
    """
    Create a FastAPI test client.
    
    This fixture provides a TestClient instance for making HTTP requests
    to the API endpoints without needing to run the actual server.
    
    Returns:
        TestClient: FastAPI test client for API requests
    """
    return TestClient(app)


@pytest.fixture
def valid_transaction():
    """
    Load valid transaction data from fraud_test.json.
    
    This fixture reads the test transaction data from the JSON file,
    providing consistent test data across multiple tests.
    
    Returns:
        dict: Transaction data with all required fields
    """
    test_data_path = Path(__file__).parent / "fraud_test.json"
    with open(test_data_path, 'r') as f:
        return json.load(f)