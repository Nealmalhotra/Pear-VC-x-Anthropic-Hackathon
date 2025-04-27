import re
from typing import List, NamedTuple

class ParsedProof(NamedTuple):
    """Represents a structured proof parsed from Claude's response."""
    theorem: str
    steps: List[str]

class ClaudeResponseParseError(ValueError):
    """Custom exception for errors during Claude response parsing."""
    pass

def parse_claude_response(text_response: str) -> ParsedProof:
    """
    Parses the raw text response from Claude to extract a structured proof.

    Assumes a format like: "THEOREM <statement> PROOF <step1>; <step2>; ... QED"
    Uses regex to extract the theorem statement and the proof steps.

    Args:
        text_response: The raw text response from the Claude API.

    Returns:
        A ParsedProof object containing the theorem and steps.

    Raises:
        ClaudeResponseParseError: If the response doesn't match the expected format.
    """
    # Normalize whitespace and case for keywords
    cleaned_response = re.sub(r'\s+', ' ', text_response).strip()
    
    # Regex to capture theorem and proof steps
    # It looks for THEOREM, captures everything until PROOF,
    # then captures everything until QED, splitting steps by ';'
    # This is a basic regex and might need refinement for complex cases.
    match = re.match(r"(?i)THEOREM\s+(.+?)\s+PROOF\s+(.+?)\s+QED", cleaned_response, re.DOTALL)
    
    if not match:
        # Fallback: Maybe Claude just gave the token list? Try previous parsing logic.
        # Note: This part is problematic as we need a *structured proof*, not tokens.
        # For now, we raise an error if the primary format isn't found.
        # A more robust solution might try multiple parsing strategies.
        raise ClaudeResponseParseError(
            f"Response does not match expected 'THEOREM ... PROOF ... QED' format. Response: {text_response[:200]}..."
        )

    theorem_statement = match.group(1).strip()
    proof_content = match.group(2).strip()
    
    # Split proof content into steps (assuming semicolon delimiter)
    # Filter out empty steps that might result from trailing semicolons or extra spaces
    proof_steps = [step.strip() for step in proof_content.split(';') if step.strip()]
    
    if not theorem_statement:
         raise ClaudeResponseParseError("Parsed theorem statement is empty.")
    # We allow empty proof steps, maybe the theorem is trivial or provable directly

    return ParsedProof(theorem=theorem_statement, steps=proof_steps)

# Example Usage (for testing)
if __name__ == '__main__':
    test_responses = [
        "THEOREM forall x exists y y = 2 * x + 1 PROOF LET z = 2 * x; LET y = z + 1 QED",
        "  theorem a > 0 proof assume a = 1; conclude a > 0 qed  ",
        "THEOREM P implies Q PROOF Assume P; Use Modus Ponens; Conclude Q QED",
        "THEOREM true PROOF QED", # Theorem with no steps
        "Invalid response format",
        "THEOREM only theorem part PROOF lacks qed",
        "PROOF only proof QED lacks theorem"
    ]

    for resp in test_responses:
        try:
            parsed = parse_claude_response(resp)
            print(f"Input: '{resp}' -> Parsed: {parsed}")
        except ClaudeResponseParseError as e:
            print(f"Input: '{resp[:50]}...' -> Error: {e}") 