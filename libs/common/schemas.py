"""
Shared data schemas for communication between microservices.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator

class TokenSequence(BaseModel):
    """
    A sequence of tokens, used across all services.
    """
    tokens: List[int] = Field(..., description="List of token IDs")
    raw_text: Optional[str] = Field(None, description="Optional original text")
    metadata: Dict[str, Any] = Field({}, description="Optional metadata about the tokens")

class NoisyTokenSequence(TokenSequence):
    """
    A sequence of tokens with noise applied.
    """
    alpha: float = Field(..., description="The noise level alpha (how much signal remains)")
    timestep: int = Field(..., description="The diffusion timestep")
    
    @validator('alpha')
    def validate_alpha(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        return v

class TokenizationRequest(BaseModel):
    """
    Request to tokenize text.
    """
    text: str = Field(..., description="The text to tokenize")
    add_special_tokens: bool = Field(True, description="Whether to add special tokens")

class TokenizationResponse(TokenSequence):
    """
    Response from tokenizing text.
    """
    pass

class NoiseRequest(BaseModel):
    """
    Request to apply noise to tokens.
    """
    tokens: List[int] = Field(..., description="List of token IDs to add noise to")
    timestep: int = Field(..., description="Current diffusion timestep")
    use_hyperschedule: bool = Field(False, description="Whether to use position-dependent noise levels")
    token_weights: Optional[List[float]] = Field(None, description="Optional weights for each token position")

class NoiseResponse(BaseModel):
    """
    Response from applying noise.
    """
    noised_tokens: List[int] = Field(..., description="Noised token IDs")
    alpha: float = Field(..., description="The noise level alpha (how much signal remains)")
    metrics: Dict[str, Any] = Field({}, description="Optional metrics about the noising process")

class RetrievalRequest(BaseModel):
    """
    Request to retrieve relevant theorems/lemmas.
    """
    query: str = Field(..., description="The query text")
    max_results: int = Field(5, description="Maximum number of results to return")

class RetrievalItem(BaseModel):
    """
    A single retrieved theorem or lemma.
    """
    id: str = Field(..., description="Unique identifier for the item")
    title: str = Field(..., description="Title or name of the theorem/lemma")
    content: str = Field(..., description="Full content of the theorem/lemma")
    relevance_score: float = Field(..., description="Relevance score from retrieval")

class RetrievalResponse(BaseModel):
    """
    Response from retrieving theorems/lemmas.
    """
    items: List[RetrievalItem] = Field(..., description="List of retrieved items")
    total_results: int = Field(..., description="Total number of results found")

class DenoiseRequest(BaseModel):
    """
    Request to denoise tokens.
    """
    noised_tokens: List[int] = Field(..., description="List of noisy token IDs")
    timestep: int = Field(..., description="Current diffusion timestep")
    context_items: Optional[List[RetrievalItem]] = Field(None, description="Optional context items from retrieval")
    guidance: Optional[str] = Field(None, description="Optional user guidance for denoising")

class DenoiseResponse(BaseModel):
    """
    Response from denoising.
    """
    denoised_tokens: List[int] = Field(..., description="Denoised token IDs")
    raw_text: str = Field(..., description="Readable denoised text")
    metrics: Dict[str, Any] = Field({}, description="Optional metrics about the denoising process")

class ProofGenerationRequest(BaseModel):
    """
    Request to generate a complete proof.
    """
    theorem_statement: str = Field(..., description="The theorem statement to prove")
    max_steps: int = Field(10, description="Maximum number of diffusion steps to perform")
    guidance: Optional[str] = Field(None, description="Optional user guidance for proof generation")

class ProofStep(BaseModel):
    """
    A single step in the proof generation process.
    """
    step_number: int = Field(..., description="Step number in the sequence")
    timestep: int = Field(..., description="Diffusion timestep")
    noised_text: str = Field(..., description="Text with noise at this step")
    denoised_text: str = Field(..., description="Denoised text at this step")
    context_items: List[RetrievalItem] = Field([], description="Context items used at this step")

class ProofGenerationResponse(BaseModel):
    """
    Response from generating a complete proof.
    """
    theorem_statement: str = Field(..., description="Original theorem statement")
    final_proof: str = Field(..., description="Final generated proof")
    steps: List[ProofStep] = Field(..., description="Steps in the proof generation process")
    metrics: Dict[str, Any] = Field({}, description="Metrics about the generation process") 