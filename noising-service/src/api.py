from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import os
import json
from .noise_schedule import TokenNoiser

# Constants
DEFAULT_VOCAB_SIZE = 50257  # Default GPT-2 vocabulary size
DEFAULT_NUM_TIMESTEPS = 1000

# Initialize FastAPI app
app = FastAPI(
    title="Noising Service API",
    description="API for applying SEDD-based noise to token sequences in the proof generation pipeline",
    version="0.1.0"
)

# Initialize the token noiser
token_noiser = None

def get_token_noiser() -> TokenNoiser:
    """
    Dependency to get or initialize the TokenNoiser instance.
    """
    global token_noiser
    if token_noiser is None:
        # Get configuration from environment variables
        vocab_size = int(os.getenv("VOCAB_SIZE", DEFAULT_VOCAB_SIZE))
        num_timesteps = int(os.getenv("NUM_TIMESTEPS", DEFAULT_NUM_TIMESTEPS))
        schedule_type = os.getenv("SCHEDULE_TYPE", "cosine")
        
        # Initialize the noiser
        token_noiser = TokenNoiser(
            vocab_size=vocab_size,
            num_timesteps=num_timesteps,
            schedule_type=schedule_type
        )
    return token_noiser

# Pydantic models for request/response validation
class NoiseRequest(BaseModel):
    tokens: List[int] = Field(..., description="List of token IDs to add noise to")
    timestep: int = Field(..., description="Current diffusion timestep (0 to num_timesteps-1)")
    use_hyperschedule: bool = Field(False, description="Whether to use position-dependent noise levels")
    token_weights: Optional[List[float]] = Field(None, description="Optional weights for each token position (for hyperschedule)")
    
    @field_validator('timestep')
    @classmethod
    def validate_timestep(cls, v):
        if v < 0 or v >= DEFAULT_NUM_TIMESTEPS:
            raise ValueError(f"Timestep must be between 0 and {DEFAULT_NUM_TIMESTEPS-1}")
        return v
    
    @field_validator('token_weights')
    @classmethod
    def validate_token_weights(cls, v, info):
        # In Pydantic v2, we need to use the info object to access other field values
        tokens = info.data.get('tokens', [])
        if v is not None and tokens and len(v) != len(tokens):
            raise ValueError("token_weights must have the same length as tokens")
        return v

class NoiseResponse(BaseModel):
    noised_tokens: List[int] = Field(..., description="Noised token IDs")
    alpha: float = Field(..., description="The noise level alpha (how much signal remains) at this timestep")
    metrics: Dict[str, Any] = Field({}, description="Optional metrics about the noising process")

# API endpoints
@app.get("/")
async def root():
    """
    Root endpoint that returns service information.
    """
    return {
        "service": "Noising Service",
        "version": "0.1.0",
        "status": "operational"
    }

@app.post("/noise", response_model=NoiseResponse)
async def apply_noise(request: NoiseRequest, noiser: TokenNoiser = Depends(get_token_noiser)):
    """
    Apply noise to a sequence of tokens according to the SEDD schedule.
    """
    try:
        # Apply noise to the token sequence
        noised_tokens = noiser.apply_noise(
            tokens=request.tokens,
            timestep=request.timestep,
            use_hyperschedule=request.use_hyperschedule,
            token_weights=request.token_weights
        )
        
        # Get noise level (alpha) for this timestep
        alpha = noiser.get_noise_level(request.timestep)
        
        # Calculate some basic metrics
        unchanged_tokens = sum(1 for i, t in enumerate(noised_tokens) if t == request.tokens[i])
        percent_unchanged = unchanged_tokens / len(request.tokens) * 100 if request.tokens else 0
        
        # Return the response
        return NoiseResponse(
            noised_tokens=noised_tokens,
            alpha=alpha,
            metrics={
                "unchanged_tokens": unchanged_tokens,
                "percent_unchanged": percent_unchanged,
                "total_tokens": len(request.tokens)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying noise: {str(e)}")

@app.get("/info")
async def get_info(noiser: TokenNoiser = Depends(get_token_noiser)):
    """
    Get information about the noising service configuration.
    """
    return {
        "vocab_size": noiser.vocab_size,
        "num_timesteps": noiser.noise_schedule.num_timesteps,
        "schedule_type": noiser.noise_schedule.schedule_type,
        "beta_min": noiser.noise_schedule.beta_min,
        "beta_max": noiser.noise_schedule.beta_max
    }

@app.get("/alpha/{timestep}")
async def get_alpha(timestep: int, noiser: TokenNoiser = Depends(get_token_noiser)):
    """
    Get the alpha value (noise level) for a specific timestep.
    """
    try:
        alpha = noiser.get_noise_level(timestep)
        return {"timestep": timestep, "alpha": alpha}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
