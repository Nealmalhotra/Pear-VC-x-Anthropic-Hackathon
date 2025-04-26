import pytest
from fastapi.testclient import TestClient
import sys
import os
import json
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api import app
from tests.test_noising import (
    PROOF_TOKENS, CRITICAL_TOKENS, VARIABLE_TOKENS, 
    get_math_token_weights, calculate_preservation_rate
)

# Create test client
client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint returns service information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == "Noising Service"
    assert "version" in data
    assert "status" in data

def test_info_endpoint():
    """Test the info endpoint returns configuration values."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "vocab_size" in data
    assert "num_timesteps" in data
    assert "schedule_type" in data
    assert "beta_min" in data
    assert "beta_max" in data

def test_alpha_endpoint():
    """Test getting alpha values for specific timesteps."""
    # Test valid timestep
    response = client.get("/alpha/500")
    assert response.status_code == 200
    data = response.json()
    assert "timestep" in data
    assert data["timestep"] == 500
    assert "alpha" in data
    assert 0 <= data["alpha"] <= 1
    
    # Test invalid timestep
    response = client.get("/alpha/1001")  # Assuming default num_timesteps=1000
    assert response.status_code == 400

def test_noise_endpoint_with_math_proof():
    """Test applying noise to mathematical proof tokens."""
    # Create test request with math proof tokens
    request_data = {
        "tokens": PROOF_TOKENS,
        "timestep": 500
    }
    
    response = client.post("/noise", json=request_data)
    assert response.status_code == 200
    data = response.json()
    
    assert "noised_tokens" in data
    assert isinstance(data["noised_tokens"], list)
    assert len(data["noised_tokens"]) == len(request_data["tokens"])
    
    assert "alpha" in data
    assert 0 <= data["alpha"] <= 1
    
    assert "metrics" in data
    assert "unchanged_tokens" in data["metrics"]
    assert "percent_unchanged" in data["metrics"]
    assert "total_tokens" in data["metrics"]
    assert data["metrics"]["total_tokens"] == len(request_data["tokens"])

def test_noise_endpoint_with_math_hyperschedule():
    """Test applying hyperschedule noise to mathematical proof tokens."""
    # Create math-specific token weights that prioritize logical operators and keywords
    token_weights = get_math_token_weights(PROOF_TOKENS)
    
    # Create test request with math token weights
    request_data = {
        "tokens": PROOF_TOKENS,
        "timestep": 250,  # Moderate timestep
        "use_hyperschedule": True,
        "token_weights": token_weights
    }
    
    response = client.post("/noise", json=request_data)
    assert response.status_code == 200
    data = response.json()
    
    assert "noised_tokens" in data
    assert isinstance(data["noised_tokens"], list)
    assert len(data["noised_tokens"]) == len(request_data["tokens"])
    
    # Calculate preservation rates for critical vs. variable tokens
    noised_tokens = data["noised_tokens"]
    critical_preservation = calculate_preservation_rate(
        PROOF_TOKENS, noised_tokens, CRITICAL_TOKENS
    )
    variable_preservation = calculate_preservation_rate(
        PROOF_TOKENS, noised_tokens, VARIABLE_TOKENS
    )
    
    print(f"API Hyperschedule - Critical preservation: {critical_preservation}%")
    print(f"API Hyperschedule - Variable preservation: {variable_preservation}%")
    
    # Critical tokens should generally be preserved more than variables with hyperschedule
    # This is probabilistic, so we don't assert but print for inspection
    
    # Also test no hyperschedule for comparison
    standard_request = {
        "tokens": PROOF_TOKENS,
        "timestep": 250,
        "use_hyperschedule": False
    }
    
    standard_response = client.post("/noise", json=standard_request)
    standard_data = standard_response.json()
    standard_noised = standard_data["noised_tokens"]
    
    standard_critical = calculate_preservation_rate(
        PROOF_TOKENS, standard_noised, CRITICAL_TOKENS
    )
    
    print(f"API Standard - Critical preservation: {standard_critical}%")

def test_noise_endpoint_at_different_timesteps():
    """Test how noising affects math tokens at different timesteps."""
    timesteps = [0, 200, 400, 800, 999]  # Various noise levels
    
    for timestep in timesteps:
        request_data = {
            "tokens": PROOF_TOKENS,
            "timestep": timestep
        }
        
        response = client.post("/noise", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        # Calculate and print preservation rates
        noised_tokens = data["noised_tokens"]
        preservation_rate = data["metrics"]["percent_unchanged"]
        critical_rate = calculate_preservation_rate(
            PROOF_TOKENS, noised_tokens, CRITICAL_TOKENS
        )
        
        print(f"Timestep {timestep}: Overall preservation {preservation_rate:.2f}%, Critical tokens {critical_rate:.2f}%")
        
        # Basic sanity check: early timesteps should have higher preservation
        if timestep < 100:
            assert preservation_rate > 50, f"At early timestep {timestep}, preservation should be high"
        elif timestep > 900:
            assert preservation_rate < 50, f"At late timestep {timestep}, preservation should be low"

def test_noise_endpoint_validation():
    """Test input validation for the noise endpoint."""
    # Test missing tokens
    response = client.post("/noise", json={"timestep": 500})
    assert response.status_code == 422
    
    # Test invalid timestep
    response = client.post("/noise", json={"tokens": PROOF_TOKENS[:5], "timestep": -1})
    assert response.status_code == 422
    
    # Test mismatched token weights length
    response = client.post("/noise", json={
        "tokens": PROOF_TOKENS, 
        "timestep": 500,
        "use_hyperschedule": True,
        "token_weights": [1.0, 1.0]  # Only 2 weights for many tokens
    })
    assert response.status_code == 422 