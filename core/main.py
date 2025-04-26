import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv, find_dotenv
import re # For extracting text between markers
import json # To load the tokenizer file
from tokenizers import Tokenizer # Use the actual library

# Import schema updates
from core.schemas import DenoiseRequest, DenoiseResponse # Removed ValidationResult
from core.retrieval_client import get_relevant_lemmas
# Import might be unused now
# from core.prompts import build_prompt, Z3_TRANSLATION_PROMPT_TEMPLATE
from core.claude_client import call_claude_api
from utils.noisifier import add_noise_to_text # Import the new noisifier
# Removed unused parser/validator imports

# --- Tokenizer Loading and Implementation ---

TOKENIZER_PATH = "math_tokenizer.json"
_tokenizer: Optional[Tokenizer] = None

def load_tokenizer():
    """Loads the tokenizer from the JSON file using the tokenizers library."""
    global _tokenizer
    try:
        _tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        # Optional: Configure truncation/padding if needed globally, 
        # otherwise configure per call if necessary.
        # _tokenizer.enable_truncation(...) 
        # _tokenizer.enable_padding(...)
        print(f"Tokenizer loaded successfully from {TOKENIZER_PATH}. Vocab size: {_tokenizer.get_vocab_size()}")
    except FileNotFoundError:
        print(f"[Error] Tokenizer file not found at {TOKENIZER_PATH}")
        raise
    except Exception as e: # Catch broader exceptions during load
        print(f"[Error] Failed to load tokenizer from {TOKENIZER_PATH}: {e}")
        raise

# Load tokenizer when the module is loaded
load_tokenizer()

def decode_tokens_to_text(tokens: List[int]) -> str:
    """Decodes a list of token IDs back into a string using the loaded tokenizer."""
    if _tokenizer is None:
        raise RuntimeError("Tokenizer is not loaded.")
    return _tokenizer.decode(tokens)

def encode_text_to_tokens(text: str) -> List[int]:
    """Encodes a string into a list of token IDs using the loaded tokenizer."""
    if _tokenizer is None:
        raise RuntimeError("Tokenizer is not loaded.")
    # The .ids retrieves the list of integer IDs from the Encoding object
    return _tokenizer.encode(text).ids

# --- End Tokenizer --- 

# Load environment variables from .env file
load_dotenv(find_dotenv())

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
RETRIEVAL_SERVICE_URL = os.getenv("RETRIEVAL_SERVICE_URL")

app = FastAPI()

# Modified flow: Receive clean text -> Add Noise -> Retrieve -> Prompt -> Call Claude -> Extract -> Encode -> Return
@app.post("/noise_and_denoise", response_model=DenoiseResponse) # Changed path to reflect action
async def noise_and_denoise_text(request: DenoiseRequest) -> DenoiseResponse:
    """Applies noise to clean text, then retrieves lemmas, prompts Claude for cleaning, encodes result."""
    print(f"Received request: clean_text='{request.clean_text[:50]}...', noise_level={request.noise_level}")

    # 1. Add Noise (New Step)
    try:
        # Using a fixed seed for reproducibility during testing/debugging, remove or make optional later
        noisy_text = add_noise_to_text(request.clean_text, noise_level=request.noise_level, seed=42) 
        print(f"Applied noise (level {request.noise_level}): '{noisy_text[:100]}...'")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[Error] Failed during text noisification: {e}")
        raise HTTPException(status_code=500, detail=f"Failed during text noisification: {e}")

    # 2. Extract the Subgoal or Theorem (Using full noisy text for now)
    retrieval_query_text = noisy_text # Use the newly generated noisy text

    # 3. Context Retrieval (RAG)
    try:
        retrieved_lemmas = await get_relevant_lemmas(retrieval_query_text, top_k=request.top_k)
        print(f"Retrieved {len(retrieved_lemmas)} lemmas based on noisy text.")
    except Exception as e:
        print(f"[Error] Failed during lemma retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Failed during lemma retrieval: {e}")

    # 4. Prompt Assembly
    lemmas_text_block = "\n".join([
        f"- {lemma.get('text', 'No text available')}"
        for lemma in retrieved_lemmas
    ])
    if not retrieved_lemmas:
        lemmas_text_block = "No relevant lemmas found or provided."

    # Construct the prompt using the internally generated noisy text
    prompt = f"""You are a proof‐repair assistant. Your task is to reconstruct a potentially corrupted mathematical proof fragment.

**Instructions:**
- Analyze the corrupted fragment below, considering the provided noise level and relevant lemmas.
- Produce the most plausible corrected proof fragment based on the input.
- **CRITICAL:** Output *only* the corrected fragment. Start your response *immediately* after the `===BEGIN_CLEAN===` marker below.
- Do **NOT** include any explanations, apologies, or text before `===BEGIN_CLEAN===` or after `===END_CLEAN===`.
- If the input is too corrupted to be certain, provide your best possible reconstruction attempt within the markers.

Corrupted proof fragment (noise level α={request.noise_level}):
"{noisy_text}"

Relevant lemmas:
{lemmas_text_block}

Example 1:
Noisy: "Assüm ther# are finit∃ly many pr1mes."
Clean: "Assume there are only finitely many primes p₁, p₂, …, pₙ."

Now, provide the corrected proof fragment. Remember, only output the fragment itself after the marker:
===BEGIN_CLEAN===""" # Ensure no trailing spaces or newlines here

    print(f"Built prompt for Claude (length: {len(prompt)})...")

    # 5. Call Claude (1st Call: Denoising)
    try:
        claude_response_text_denoise = await call_claude_api(prompt)
        print(f"Raw denoising response from Claude API: {claude_response_text_denoise[:500]}...")

        # Extract cleaned text (theorem statement)
        match = re.search(r"===BEGIN_CLEAN===(.*?)(?:===END_CLEAN===|$)", claude_response_text_denoise, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_theorem_statement = match.group(1).strip()
            print(f"Extracted cleaned theorem statement: '{cleaned_theorem_statement}'")
            if not cleaned_theorem_statement:
                 raise ValueError("Extracted cleaned theorem statement is empty.")
        else:
            raise ValueError("Could not extract cleaned theorem statement between BEGIN_CLEAN/END_CLEAN markers.")

    except HTTPException as e:
        raise e # Re-raise HTTP exceptions from Claude call
    except ValueError as e:
        print(f"[Error] Failed to extract cleaned theorem statement: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract cleaned theorem statement from Claude response: {e}")
    except Exception as e:
        print(f"[Error] Unexpected error during Claude denoising call or extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during Claude denoising call or extraction: {e}")

    # --- NEW: Proof Generation Stage ---
    
    # 6. Build Prompt for Proof Generation
    # Reuse lemmas from retrieval based on noisy text? Or retrieve again based on clean text?
    # Let's reuse for now.
    proof_prompt = f"""You are an expert mathematical assistant.
Your task is to generate a step-by-step proof for the following theorem statement.

Theorem Statement:
{cleaned_theorem_statement}

Relevant Contextual Information (Lemmas/Theorems):
{lemmas_text_block}

Please provide a clear, logical, step-by-step proof for the theorem.
Format the proof clearly, with each step numbered or on a new line.

Proof:
""" # Guide Claude to start the proof
    print(f"Built proof generation prompt (length: {len(proof_prompt)})...")

    # 7. Call Claude (2nd Call: Proof Generation)
    try:
        claude_response_text_proof = await call_claude_api(proof_prompt)
        # Assume the response *is* the proof body
        generated_proof_steps = claude_response_text_proof.strip()
        print(f"Raw proof response from Claude API: {generated_proof_steps[:500]}...")
        if not generated_proof_steps:
            raise ValueError("Claude returned an empty proof.")
            
    except HTTPException as e:
        # Specific handling for errors during the *second* call
        print(f"[Error] HTTP error during proof generation call: {e.detail}")
        raise HTTPException(status_code=500, detail=f"Claude call failed during proof generation: {e.detail}")
    except ValueError as e:
         print(f"[Error] {e}")
         raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"[Error] Unexpected error during proof generation call: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during proof generation: {e}")

    # 8. Format the Final Output
    formatted_proof = ( 
        f"Theorem:\n{cleaned_theorem_statement}\n\n" 
        f"Proof:\n{generated_proof_steps}\n\nQ.E.D."
    )

    # 9. Return (Updated Response Schema)
    return DenoiseResponse(
        formatted_proof=formatted_proof,
        debug={
            "original_clean_text": request.clean_text,
            "applied_noise_level": request.noise_level,
            "generated_noisy_text": noisy_text,
            "retrieval_queries": [retrieval_query_text],
            "retrieved_lemmas_count": len(retrieved_lemmas),
            "denoising_prompt_length": len(prompt),
            "raw_denoising_response_preview": claude_response_text_denoise[:200],
            "cleaned_theorem_statement": cleaned_theorem_statement,
            "proof_generation_prompt_length": len(proof_prompt),
            "raw_proof_response_preview": generated_proof_steps[:200]
        }
    )

# Remove the old endpoint and root endpoint if no longer needed
# @app.post("/denoise_and_validate", response_model=DenoiseResponse)
# async def denoise_and_validate(request: DenoiseRequest) -> DenoiseResponse:
# ... (old code removed for clarity) ...

@app.get("/")
async def root():
    return {"message": "Noise and Denoise Service is running"} 