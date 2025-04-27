# Stores prompt templates and building logic 
from typing import List, Dict, Any, Optional

# Basic prompt template for the first attempt
INITIAL_DENOISE_PROMPT_TEMPLATE = """
System: You are an expert in mathematical logic and theorem proving. Your task is to take a potentially noisy sequence of proof tokens and clean it up based on the provided context and noise level.
The 'Context Lemmas' below were retrieved from a Pinecone vector index containing embeddings of theorems and lemmas from the Metamath dataset (set.mm, iset.mm). They are semantically similar to the potential meaning of the noisy tokens and should help guide the reconstruction of a valid proof.

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
System: You are an expert in mathematical logic and theorem proving. Your previous attempt to clean the proof tokens failed validation. Use the feedback below and the provided context lemmas to refine your output.
The 'Context Lemmas' below were retrieved from a Pinecone vector index containing embeddings of theorems and lemmas from the Metamath dataset (set.mm, iset.mm). They are semantically similar to the potential meaning of the noisy tokens and should help guide the reconstruction of a valid proof, even after failed attempts.

Original Noisy Tokens (at noise level alpha_t={alpha_t}):
{noisy_tokens_str}

Context Lemmas:
{lemmas_text}

Previous Attempt (that failed validation):
{previous_attempt_str}

Validation Feedback:
{validation_error}

Based on the feedback and context lemmas, refine the proof. Output the result *strictly* in the following format:
THEOREM <refined_theorem_statement> PROOF <refined_step_1>; <refined_step_2>; ...; <refined_step_n> QED
Do not include *any* other text, explanations, or formatting outside this structure.

Assistant:
THEOREM""" # Start the response to guide Claude

# --- Z3 Translation Prompt ---

Z3_TRANSLATION_PROMPT_TEMPLATE = """
System: You are an expert Python programmer specializing in the Z3 theorem prover library (`z3-solver`). Your task is to translate a given mathematical theorem statement into executable Z3 Python code that defines a single Z3 assertion variable `z3_assertion`.

**VERY IMPORTANT Z3 Syntax Rules:**
- You **MUST** prefix all Z3 functions, sorts, and constants with `z3.` (e.g., `z3.And()`, `z3.ForAll()`, `z3.Int()`, `z3.BoolSort()`, `z3.IntSort()`).
- **DO NOT** use `from z3 import *`.
- **DO NOT** define intermediate Python helper functions. Construct the `z3_assertion` expression directly.
- Declare all necessary Z3 variables (like `z3.Int('x')`, `z3.Bool('p')`) *before* they are used in the `z3_assertion`.
- Use `z3.And()`, `z3.Or()`, `z3.Not()`, `z3.Implies()` for boolean logic on Z3 expressions.
- Use `z3.ForAll([var1, var2,...], body)` for universal quantification. Declare the quantified variables (`var1`, `var2`, etc.) before the `z3.ForAll` call.
- Use `z3.Exists([var1, var2,...], body)` for existential quantification. Declare the quantified variables before the `z3.Exists` call.
- Use `z3.If(condition, then_expr, else_expr)` for symbolic conditional logic.

The code should:
1. Import z3 (`import z3`).
2. Declare ALL necessary Z3 variables (constants like `N = z3.Int('N')` and quantified variables like `x = z3.Int('x')`) using the `z3.` prefix.
3. Construct the single Z3 expression for `z3_assertion` directly, using the declared variables and Z3 functions (with `z3.` prefixes).
4. Assign the final Z3 expression object to a variable named `z3_assertion`.
5. Output *only* the raw Python code. Do not include explanations, comments outside the code, markdown formatting, or any surrounding text.

Theorem Statement:
{theorem_statement}

Assistant:
```python
import z3

# Declare ALL variables (constants and quantified) using z3.
{declarations}

# Construct the single z3_assertion expression directly
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