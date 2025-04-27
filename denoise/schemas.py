from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class DenoiseRequest(BaseModel):
    # noisy_tokens: List[int] # Changed: Now accepting clean text
    clean_text: str = Field(..., description="The original clean text to be noised and then denoised.")
    # alpha_t: float = Field(..., ge=0.0, le=1.0, description="Noise level alpha_t") # Renamed
    noise_level: float = Field(..., ge=0.0, le=1.0, description="Probability (0.0-1.0) of character substitution noise to apply.")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of lemmas to retrieve")

class ValidationResult(BaseModel):
    is_valid: bool
    details: Optional[str] = None

class DenoiseResponse(BaseModel):
    # clean_tokens: List[int] = Field(..., description="The cleaned sequence of tokens.")
    formatted_proof: Optional[str] = Field(None, description="Human-readable version of the generated proof, or None if unsuccessful.")
    debug: Dict[str, Any] = Field(..., description="Debug information including retrieval queries, intermediate steps.")

# class DenoiseResponse(BaseModel):
#     # final_clean_tokens: List[int] # Removing placeholder for now, focusing on formatted proof
#     formatted_proof: Optional[str] = Field(None, description="Human-readable version of the validated proof.")
#     validation_result: ValidationResult
#     iterations_taken: int
#     debug: Optional[Dict[str, Any]] = Field(None, description="Optional debug info") 