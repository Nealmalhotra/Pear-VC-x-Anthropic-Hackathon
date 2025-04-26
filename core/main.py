import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv, find_dotenv

# Import schema updates
from core.schemas import DenoiseRequest, DenoiseResponse, ValidationResult 
from core.retrieval_client import get_relevant_lemmas
# Import both prompt templates
from core.prompts import build_prompt, Z3_TRANSLATION_PROMPT_TEMPLATE 
from core.claude_client import call_claude_api
# Import the real parser and its exception
from core.parsing import parse_claude_response, ParsedProof, ClaudeResponseParseError 
# Import the real validator
from core.validator import validate_proof_with_z3

# Load environment variables from .env file
load_dotenv(find_dotenv())

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
RETRIEVAL_SERVICE_URL = os.getenv("RETRIEVAL_SERVICE_URL")

app = FastAPI()

# Hour 4: Integrate real parser and validator + Hour 4.5: Use Claude for Z3 Translation
@app.post("/denoise_and_validate", response_model=DenoiseResponse)
async def denoise_and_validate(request: DenoiseRequest) -> DenoiseResponse:
    """Iteratively attempts to denoise tokens, validates with Z3."""
    print(f"Received request: noisy_tokens={request.noisy_tokens}, alpha_t={request.alpha_t}, max_iterations={request.max_iterations}")
    
    current_retrieval_tokens = request.noisy_tokens # Tokens used for lemma retrieval
    original_noisy_tokens = request.noisy_tokens # Keep original for feedback prompt
    last_claude_response_text = "" 
    last_parsed_proof: Optional[ParsedProof] = None # Store the ParsedProof object
    last_validation_result = ValidationResult(is_valid=False, details="Loop did not run or first attempt failed parsing/validation.")
    last_z3_code_str: Optional[str] = None # Store the generated Z3 code

    for i in range(request.max_iterations):
        print(f"\n--- Iteration {i + 1}/{request.max_iterations} ---")
        
        # 1. Call retrieval service
        retrieved_lemmas = await get_relevant_lemmas(current_retrieval_tokens)
        print(f"Retrieved lemmas (dummy): {retrieved_lemmas}")

        # 2. Build prompt (Pass feedback if available from previous iteration)
        # Convert last_parsed_proof back to some token representation if needed for feedback prompt?
        # The FEEDBACK_DENOISE_PROMPT_TEMPLATE expects previous_attempt_tokens: List[int]
        # Our parser now returns ParsedProof. We need to adapt the prompt or add a step 
        # to convert ParsedProof back to tokens for the feedback mechanism.
        # TODO: Revisit feedback mechanism with structured proof
        previous_attempt_token_placeholder = None # Placeholder until feedback mechanism is updated
        prompt = build_prompt(
            noisy_tokens=original_noisy_tokens, 
            alpha_t=request.alpha_t, 
            lemmas=retrieved_lemmas,
            previous_attempt_tokens=previous_attempt_token_placeholder, # Use placeholder
            validation_error=last_validation_result.details if not last_validation_result.is_valid and i > 0 else None
        )
        print(f"Built prompt for Claude... (length: {len(prompt)}) {'with feedback' if previous_attempt_token_placeholder else 'initial'}")

        # 3. Call Claude API
        try:
            claude_response_text = await call_claude_api(prompt)
            last_claude_response_text = claude_response_text
            print(f"Raw response from Claude API: {claude_response_text[:200]}...")
        except HTTPException as e:
            raise e
        except Exception as e:
            print(f"Unexpected error during Claude call: {e}")
            raise HTTPException(status_code=500, detail=f"Error in iteration {i+1} during Claude call: {e}")

        # 4. Parse Claude response (Using real parser)
        try:
            parsed_proof = parse_claude_response(claude_response_text)
            last_parsed_proof = parsed_proof
            print(f"Parsed proof: Theorem='{parsed_proof.theorem}', Steps={parsed_proof.steps}")
        except ClaudeResponseParseError as e:
             print(f"[Error] Failed to parse Claude response: {e}")
             last_validation_result = ValidationResult(is_valid=False, details=f"Parsing Error: {e}")
             # If parsing fails, we cannot validate. Continue to next iteration or break?
             if i == request.max_iterations - 1:
                 print("Max iterations reached, last attempt failed parsing.")
                 # Update final result details
                 last_validation_result = ValidationResult(is_valid=False, details=f"Max iterations reached. Last attempt failed parsing: {e}")
             continue # Skip validation and proceed to next iteration (or exit)

        # 4.5 NEW STEP: Call Claude to translate theorem to Z3 Python code
        try:
            translation_prompt = Z3_TRANSLATION_PROMPT_TEMPLATE.format(
                theorem_statement=parsed_proof.theorem,
                # The placeholders in the assistant part are guides for Claude,
                # not for us to fill here.
                declarations="", 
                assertion_code=""
            )
            print(f"Building Z3 translation prompt for: {parsed_proof.theorem}")
            z3_code_str = await call_claude_api(translation_prompt)
            # Basic cleaning: remove potential markdown backticks
            z3_code_str = z3_code_str.strip().removeprefix("```python").removesuffix("```").strip()
            last_z3_code_str = z3_code_str # Store for debug/final response
            print(f"Received Z3 code string (raw):\n---\n{z3_code_str}\n---")
        except HTTPException as e:
            # Re-raise specific HTTP errors if needed, or handle generally
            print(f"[Error] HTTP error during Z3 translation call: {e.detail}")
            last_validation_result = ValidationResult(is_valid=False, details=f"HTTP Error during Z3 translation: {e.detail}")
            if i == request.max_iterations - 1:
                 print("Max iterations reached, last attempt failed Z3 translation.")
            continue # Skip validation, proceed to next iteration or exit
        except Exception as e:
            print(f"[Error] Unexpected error during Z3 translation call: {e}")
            last_validation_result = ValidationResult(is_valid=False, details=f"Unexpected Error during Z3 translation: {e}")
            if i == request.max_iterations - 1:
                 print("Max iterations reached, last attempt failed Z3 translation.")
            continue # Skip validation, proceed to next iteration or exit
        
        # 5. Validate with Z3 (Using the generated code string)
        # The validator function will be updated to accept this string
        validation_result = validate_proof_with_z3(z3_code_str)
        last_validation_result = validation_result
        print(f"Validation result: {validation_result}")

        # 6. Check validation and decide next step
        if validation_result.is_valid:
            print(f"Validation SUCCESSFUL in iteration {i + 1}!")
            # TODO: Implement structured_proof_to_tokens conversion
            final_clean_tokens_placeholder = [0] # Placeholder
            return DenoiseResponse(
                final_clean_tokens=final_clean_tokens_placeholder,
                validation_result=validation_result,
                iterations_taken=i + 1,
                debug={
                    "final_parsed_proof": f"{parsed_proof}",
                    "final_z3_code": last_z3_code_str
                    }
            )
        else:
            print(f"Validation FAILED in iteration {i + 1}.")
            # Decide how to update current_retrieval_tokens for the next iteration?
            # Option 1: Keep original (current implementation implicitly does this)
            # Option 2: Try to get tokens from last_parsed_proof?
            # Keeping original seems safer for now.
            if i == request.max_iterations - 1:
                 print("Max iterations reached without a valid proof.")
                 # Final result details already set in last_validation_result

    # If loop finishes without valid proof (due to validation failure or parse error on last iteration)
    # Return the tokens from the last *parsed* attempt, or original if none parsed.
    # TODO: Implement structured_proof_to_tokens conversion for this return too.
    final_clean_tokens_placeholder = [0] # Placeholder
    if last_parsed_proof:
        # final_clean_tokens_placeholder = structured_proof_to_tokens(last_parsed_proof)
        pass # Keep placeholder for now
    else:
        final_clean_tokens_placeholder = request.noisy_tokens # Fallback to original noisy
        
    return DenoiseResponse(
        final_clean_tokens=final_clean_tokens_placeholder, 
        validation_result=last_validation_result, 
        iterations_taken=request.max_iterations,
        debug={
            "last_claude_response": last_claude_response_text[:500],
            "last_parsed_proof": f"{last_parsed_proof}" if last_parsed_proof else "None",
            "last_z3_code": last_z3_code_str if last_z3_code_str else "None",
            }
    )

@app.get("/")
async def root():
    return {"message": "Denoiser Service with Validation Loop is running"} 