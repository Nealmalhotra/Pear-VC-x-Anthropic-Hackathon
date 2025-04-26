# Stores prompt templates and building logic 
from typing import List, Dict, Any, Optional

# Basic prompt template for the first attempt
INITIAL_DENOISE_PROMPT_TEMPLATE = """
System: You are an expert in mathematical logic and theorem proving. Your task is to take a potentially noisy sequence of proof tokens and clean it up based on the provided context and noise level.

Context Lemmas:
{lemmas_text}

Noisy Tokens (at noise level alpha_t={alpha_t}):
{noisy_tokens_str}

Clean the noisy tokens and formulate a structured proof. Output the result *strictly* in the following format:
THEOREM <theorem_statement> PROOF <step_1>; <step_2>; ...; <step_n> QED
Do not include *any* other text, explanations, or formatting outside this structure.

Assistant:
THEOREM""" # Start the response to guide Claude

# Template for subsequent attempts, incorporating feedback from validation
FEEDBACK_DENOISE_PROMPT_TEMPLATE = """
System: You are an expert in mathematical logic and theorem proving. Your previous attempt to clean the proof tokens failed validation. Use the feedback below to refine your output.

Original Noisy Tokens (at noise level alpha_t={alpha_t}):
{noisy_tokens_str}

Context Lemmas:
{lemmas_text}

Previous Attempt (that failed validation):
{previous_attempt_str}

Validation Feedback:
{validation_error}

Based on the feedback, refine the proof. Output the result *strictly* in the following format:
THEOREM <refined_theorem_statement> PROOF <refined_step_1>; <refined_step_2>; ...; <refined_step_n> QED
Do not include *any* other text, explanations, or formatting outside this structure.

Assistant:
THEOREM""" # Start the response to guide Claude

# --- Z3 Translation Prompt ---

Z3_TRANSLATION_PROMPT_TEMPLATE = """
System: You are an expert Python programmer specializing in the Z3 theorem prover library (`z3-solver`). Your task is to translate a given mathematical theorem statement into executable Z3 Python code.

The code should:
1. Declare any necessary Z3 variables (e.g., `x = z3.Int('x')`) or functions (e.g., `P = z3.Function('P', z3.IntSort(), z3.BoolSort())`). Assume standard sorts like IntSort and BoolSort unless specified otherwise. Default variables to IntSort.
2. Construct the Z3 assertion corresponding to the theorem statement.
3. Assign the final Z3 assertion object to a variable named `z3_assertion`.
4. Output *only* the raw Python code. Do not include explanations, comments outside the code, markdown formatting, or any surrounding text.

Theorem Statement:
{theorem_statement}

Assistant:
```python
import z3

# Declare variables and functions
{declarations}

# Construct the assertion
z3_assertion = {assertion_code}
```""" 

def build_prompt(noisy_tokens: List[int], alpha_t: float, lemmas: List[Dict[str, Any]], previous_attempt_tokens: Optional[List[int]] = None, validation_error: Optional[str] = None) -> str:
    """Builds the prompt for Claude, using a different template if validation feedback is provided."""
    
    lemmas_text = "\n".join([
        f"- ID: {lemma.get('id', 'N/A')}, Score: {lemma.get('score', 'N/A')}\n  Text: {lemma.get('text', 'No text available')}"
        for lemma in lemmas
    ])
    if not lemmas:
        lemmas_text = "No relevant lemmas provided."

    noisy_tokens_str = ' '.join(map(str, noisy_tokens))

    if previous_attempt_tokens is not None and validation_error is not None:
        # Use the feedback template
        previous_attempt_str = ' '.join(map(str, previous_attempt_tokens))
        prompt = FEEDBACK_DENOISE_PROMPT_TEMPLATE.format(
            lemmas_text=lemmas_text,
            alpha_t=alpha_t,
            noisy_tokens_str=noisy_tokens_str,
            previous_attempt_str=previous_attempt_str,
            validation_error=validation_error
        )
    else:
        # Use the initial template
        prompt = INITIAL_DENOISE_PROMPT_TEMPLATE.format(
            lemmas_text=lemmas_text,
            alpha_t=alpha_t,
            noisy_tokens_str=noisy_tokens_str
        )
    
    return prompt 