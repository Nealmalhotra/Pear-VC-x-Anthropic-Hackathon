from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import os
import json
from tokenizer import MathBPETokenizer
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize the app
app = FastAPI(
    title="Math BPE Tokenizer Service",
    description="A service to tokenize mathematical text with BPE algorithm",
    version="1.0.0",
)

# Load or initialize the tokenizer
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "data/math_tokenizer.json")

# Initialize global tokenizer
tokenizer = None

# Input model
class TokenizeRequest(BaseModel):
    text: str = Field(..., description="The text to tokenize")
    return_text_tokens: bool = Field(False, description="Whether to return token text along with IDs")

# Output model
class TokenizeResponse(BaseModel):
    tokens: List[int] = Field(..., description="List of token IDs")
    tokens_text: Optional[List[str]] = Field(None, description="List of token text (if requested)")

@app.on_event("startup")
async def startup_event():
    """Load or initialize the tokenizer when the app starts."""
    global tokenizer
    try:
        if os.path.exists(TOKENIZER_PATH):
            logger.info(f"Loading tokenizer from {TOKENIZER_PATH}")
            tokenizer = MathBPETokenizer.from_file(TOKENIZER_PATH)
        else:
            logger.warning(f"Tokenizer not found at {TOKENIZER_PATH}, initializing a new one")
            # For a new tokenizer, we should train it, but here we'll just initialize
            tokenizer = MathBPETokenizer()
            
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
            
            # Sample math texts to train the tokenizer (in production, use a larger corpus)
            sample_texts = [
                "Let x be a real number.",
                "For all epsilon > 0, there exists delta > 0 such that...",
                "The integral of f(x) from a to b is...",
                "\\sum_{i=1}^{n} i^2 = \\frac{n(n+1)(2n+1)}{6}",
                "If A and B are sets, then A \\cup B = \\{x | x \\in A \\text{ or } x \\in B\\}",
                "Prove that there are infinitely many prime numbers.",
                "Let P(n) be the statement that n^2 + n + 41 is prime for all natural numbers n.",
                "Consider a sequence defined by a_n = a_{n-1} + a_{n-2} with a_0 = 0 and a_1 = 1."
            ]
            tokenizer.train(sample_texts, save_path=TOKENIZER_PATH)
    except Exception as e:
        logger.error(f"Error initializing tokenizer: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

@app.post("/tokenize", response_model=TokenizeResponse, tags=["Tokenization"])
async def tokenize_text(request: TokenizeRequest):
    """
    Tokenize input text and return token IDs.
    Optionally returns the token text as well.
    """
    try:
        if tokenizer is None:
            raise HTTPException(status_code=500, detail="Tokenizer not initialized")
        
        result = tokenizer.encode(request.text)
        
        response = {"tokens": result["tokens"]}
        if request.return_text_tokens:
            response["tokens_text"] = result["tokens_text"]
        else:
            response["tokens_text"] = None
            
        return response
    except Exception as e:
        logger.error(f"Error processing tokenize request: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health", tags=["Health"])
async def health_check():
    """Check if the service is healthy."""
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not initialized")
    return {"status": "healthy", "tokenizer_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 