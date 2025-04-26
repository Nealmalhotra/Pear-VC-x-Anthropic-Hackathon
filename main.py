import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

# Assuming schemas.py exists
from schemas import DenoiseRequest, DenoiseResponse 
# Assuming retrieval_client.py exists
from retrieval_client import get_relevant_lemmas 

# Load environment variables from .env file
load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
RETRIEVAL_SERVICE_URL = os.getenv("RETRIEVAL_SERVICE_URL")

app = FastAPI()

@app.post("/denoise", response_model=DenoiseResponse)
async def denoise(request: DenoiseRequest) -> DenoiseResponse:
    """Receives noisy tokens and alpha_t, retrieves lemmas, returns dummy cleaned tokens."""
    print(f"Received request: noisy_tokens={request.noisy_tokens}, alpha_t={request.alpha_t}")
    # print(f"Using Claude Key: {CLAUDE_API_KEY[:5]}..." if CLAUDE_API_KEY else "Claude Key not set") # User removed this
    print(f"Using Retrieval Service URL: {RETRIEVAL_SERVICE_URL}")

    # Call the (mocked) retrieval service
    retrieved_lemmas = await get_relevant_lemmas(request.noisy_tokens)
    print(f"Retrieved lemmas (dummy): {retrieved_lemmas}")

    # --- Integration step starts here ---
    # TODO: Build prompt (using prompts.build_prompt)
    # TODO: Call Claude API (using claude_client.call_claude_api)
    # TODO: Parse response (Hour 4)
    # ------------------------------------

    # Dummy response logic (unchanged for now)
    dummy_cleaned_tokens = [token + 1 for token in request.noisy_tokens] # Simple dummy logic

    return DenoiseResponse(clean_tokens=dummy_cleaned_tokens)

# Optional: Add a root endpoint for basic health check
@app.get("/")
async def root():
    return {"message": "Denoiser Service is running"} 