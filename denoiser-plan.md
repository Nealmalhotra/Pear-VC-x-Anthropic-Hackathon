# Development Plan: Reverse Denoiser & Prompt Orchestrator (6-Hour Hackathon with Z3 Validation Loop)

This plan outlines the steps to build the core functionality of the Reverse Denoiser & Prompt Orchestrator service within a 6-hour timeframe, **including a validation loop using the Z3 theorem prover**. The focus is on delivering an MVP that attempts to denoise and validate proofs iteratively.

**Framework Choice:** FastAPI.
**Key Libraries:** `fastapi`, `uvicorn`, `pydantic`, `httpx`, `python-dotenv`, `anthropic`, `z3-solver` (for Z3).

**Assumption:** The proof representation (after parsing Claude's output) can be converted into a format Z3 understands (e.g., SMT-LIB or direct Z3 Python API usage).

---

**Hour 1: Project Setup & API Skeleton**

1.  **Initialize Project:**
    *   Create project directory (`denoiser_service`).
    *   Set up virtual environment (`python -m venv venv && source venv/bin/activate`).
    *   Create `main.py`, `requirements.txt`, `.env`, `.gitignore`.
    *   Initialize Git (`git init`).
2.  **Install Dependencies:**
    *   Add `fastapi`, `uvicorn[standard]`, `pydantic`, `httpx`, `python-dotenv`, `anthropic`, `z3-solver` to `requirements.txt`.
    *   Run `pip install -r requirements.txt`.
3.  **Define API Models (Pydantic):**
    *   In `main.py` (or `schemas.py`):
        *   `DenoiseRequest` (`noisy_tokens: List[int]`, `alpha_t: float`, `max_iterations: Optional[int] = 3`)
        *   `ValidationResult` (`is_valid: bool`, `details: Optional[str] = None`)
        *   `DenoiseResponse` (`final_clean_tokens: List[int]`, `validation_result: ValidationResult`, `iterations_taken: int`, `debug: Optional[dict] = None`)
4.  **Create FastAPI App & Endpoint:**
    *   In `main.py`, create the FastAPI app instance.
    *   Define `POST /denoise_and_validate`.
    *   Use Pydantic models for request/response. Basic input validation via FastAPI/Pydantic.
5.  **Dummy Response:**
    *   Make `/denoise_and_validate` return a hardcoded dummy `DenoiseResponse`.
    *   *Goal: Run `uvicorn main:app --reload`, hit endpoint successfully.*
6.  **Environment Variables Setup:**
    *   Add placeholders for `CLAUDE_API_KEY`, `RETRIEVAL_SERVICE_URL` in `.env`. Load them.

**Hour 2: Retrieval & Basic Loop Structure**

1.  **Create Retrieval Client Function:**
    *   Async `get_relevant_lemmas(tokens)` using `httpx`. Mock return data initially.
2.  **Z3 Validator Stub:**
    *   Create a function `validate_proof_with_z3(proof_representation)` that takes the parsed proof.
    *   **Crucially, for now, make this return a dummy `ValidationResult(is_valid=False)` to allow loop testing.** Actual Z3 logic comes later.
3.  **Implement Basic Loop in Endpoint:**
    *   Inside `/denoise_and_validate`:
        *   Initialize `current_tokens = request.noisy_tokens`.
        *   Start a loop (`for i in range(request.max_iterations)`).
        *   Call `get_relevant_lemmas` (mocked).
        *   *(Placeholder for Claude call)*
        *   *(Placeholder for Parsing)*
        *   Call `validate_proof_with_z3` (stubbed).
        *   If `is_valid` is true, break the loop.
        *   *(Placeholder for feeding result back to next iteration)*
    *   Return the final result (still dummy tokens) and validation status.
    *   *Goal: Endpoint simulates the loop structure using mocked/stubbed functions.*

**Hour 3: Prompt Construction & Claude API Call**

1.  **Define Prompt Template(s):**
    *   Store initial denoising prompt.
    *   **Consider a secondary prompt template** for iterations *after* the first, possibly including Z3's error feedback if available/useful.
2.  **Create Prompt Building Function:**
    *   build_prompt(tokens, alpha_t, lemmas, previous_error=None)` to format the template.
3.  **Create Claude Client Function:**
    *   Async `call_claude_api(prompt)` using `anthropic`. Handle basic errors.
4.  **Integrate into Loop:**
    *   Inside the loop:
        *   Call `build_prompt` with `current_tokens`, lemmas, and potentially error info from the *previous* Z3 validation.
        *   Call `call_claude_api`.
    *   *Goal: Loop calls mocked retrieval, builds prompt, calls Claude (real call!).*

**Hour 4: Parsing & Z3 Integration (Core Logic)**

1.  **Implement Parsing Logic:**
    *   `parse_claude_response(text_response)` -> Convert Claude's text output into the **structured proof format needed for Z3**. This is a critical step.
2.  **Integrate Parsing into Loop:**
    *   After `call_claude_api`, call `parse_claude_response`. Let `parsed_proof` be the result.
3.  **Implement Z3 Validator:**
    *   Flesh out `validate_proof_with_z3(parsed_proof)`:
        *   Convert `parsed_proof` into Z3 constraints/assertions.
        *   Create a Z3 Solver instance.
        *   Add assertions.
        *   Call `solver.check()`.
        *   Return `ValidationResult(is_valid=True/False, details=...)`. Include Z3 model/unsat core info in details if possible upon failure.
4.  **Connect Loop Feedback:**
    *   Inside the loop, after `validate_proof_with_z3`:
        *   If `is_valid`, store `parsed_proof` (or its token equivalent) as the final result and break.
        *   If not valid, update `current_tokens` based on `parsed_proof` for the *next* iteration. Pass validation details (`result.details`) to the next `build_prompt` call.
    *   *Goal: The core loop runs: Claude -> Parse -> Z3 Check -> Update -> (repeat or break).*

**Hour 5: Refinement & Basic Reliability**

1.  **Implement Basic Retries (Claude):**
    *   Add simple retry logic (e.g., 3 attempts) around `call_claude_api`.
2.  **Implement In-Memory Cache:**
    *   Cache `call_claude_api` calls using `functools.lru_cache` based on the prompt hash. Configure size (`MAX_CACHE_SIZE`).
3.  **Handle Max Iterations:**
    *   Ensure the loop terminates after `max_iterations` even if validation fails. Return the last attempt's tokens and the failing validation result.
4.  **Basic Rate Limit Handling (Claude):**
    *   Catch Claude's rate limit error and return HTTP 429.
5.  **Update Response:**
    *   Ensure the final `DenoiseResponse` correctly includes `final_clean_tokens` (convert the validated proof back to tokens if needed), `validation_result`, and `iterations_taken`.
    *   *Goal: Loop has basic caching, retries, terminates correctly, and returns informative response.*

**Hour 6: Testing, Integration & Documentation**

1.  **Integration Testing (Manual/Curl):**
    *   Test the full flow with sample `noisy_tokens`.
    *   Swap out mocked retrieval (`get_relevant_lemmas`) for the real service (if available).
    *   Test cases:
        *   Valid proof found quickly.
        *   Valid proof found after several iterations.
        *   No valid proof found within `max_iterations`.
        *   Error cases (invalid input, retrieval down, Claude errors, Z3 errors).
2.  **Refine Error Handling/Messages:**
    *   Improve error messages for 500s. Handle potential errors during Z3 processing.
3.  **Check Environment Variables:**
    *   Verify `CLAUDE_API_KEY`, `RETRIEVAL_SERVICE_URL`, `MAX_CACHE_SIZE` are used. Add `MAX_ITERATIONS` default.
4.  **Code Cleanup & Comments:** Add clarity.
5.  **Update OpenAPI Docs:** Review `/docs`.
    *   *Goal: Service is runnable, handles the core validation loop, basic errors, caching, and integrates with Retrieval. Docs usable.*

---

**Out of Scope / Stretch Goals:**

*   Sophisticated Z3 error parsing for better feedback prompts.
*   Advanced retry backoff.
*   Redis caching.
*   Prometheus metrics.
*   Async Z3 calls (if solving takes time).
*   Formal API key security.
*   Automated tests.

This plan is ambitious for 6 hours, especially the Parse -> Z3 -> Feedback loop. Focus on getting the structure right and the Z3 call working, even if the feedback mechanism is simple initially. 