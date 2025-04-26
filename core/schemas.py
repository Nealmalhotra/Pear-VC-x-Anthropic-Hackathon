from typing import List, Optional, Dict
from pydantic import BaseModel

class DenoiseRequest(BaseModel):
    noisy_tokens: List[int]
    alpha_t: float
    max_iterations: Optional[int] = 3

class ValidationResult(BaseModel):
    is_valid: bool
    details: Optional[str] = None

class DenoiseResponse(BaseModel):
    final_clean_tokens: List[int]
    validation_result: ValidationResult
    iterations_taken: int
    debug: Optional[Dict] = None 