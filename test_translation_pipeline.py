import asyncio
import os
import sys
from typing import List, Dict, Any

# Add core directory to sys.path to allow imports
# Assumes this script is in the project root
project_root = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(project_root, 'core')
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

from dotenv import load_dotenv, find_dotenv

# Import necessary functions from the core modules
try:
    # Need to ensure core modules are importable from project root
    from core.retrieval_client import get_relevant_lemmas
    from core.prompts import build_prompt, Z3_TRANSLATION_PROMPT_TEMPLATE
    from core.claude_client import call_claude_api, CLAUDE_API_KEY # Check if API key is loaded
    from core.parsing import parse_claude_response, ClaudeResponseParseError
    from core.validator import validate_proof_with_z3 # Add validator import
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Ensure you are running this script from the project root directory ({project_root})")
    print(f"Looking for core modules in: {core_dir}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

async def run_denoising_test(test_case: Dict[str, Any], alpha_t: float, target_clean_readable: str):
    """
    Runs only the denoising part of the pipeline for a given test case
    and prints the comparison.
    """
    noisy_tokens = test_case['noisy_tokens']
    print(f"--- Running Test Case: {test_case['name']} ---")
    print(f"Input Noisy Tokens: {noisy_tokens}")
    print(f"Readable Noisy: {test_case.get('readable_noisy', 'N/A')}")
    print(f"Input Alpha_t: {alpha_t}")
    print(f"TARGET Clean Readable: {target_clean_readable}")
    print("-" * 30)

    # 1. Retrieval (Using dummy data)
    try:
        retrieved_lemmas = await get_relevant_lemmas(noisy_tokens)
        # print(f"Retrieved Lemmas (Dummy): {retrieved_lemmas}") # Less relevant for this test
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return
    # print("-" * 30)

    # 2. Build Denoising Prompt
    denoise_prompt = build_prompt(
        noisy_tokens=noisy_tokens,
        alpha_t=alpha_t,
        lemmas=retrieved_lemmas
    )

    # 3. Call Claude for Denoising/Proof Generation
    denoised_proof_str = None
    try:
        print("Calling Claude for Denoising/Proof Generation...")
        denoised_proof_str = await call_claude_api(denoise_prompt)
        print("==> ACTUAL Received Denoised Proof String:")
        print(f"```\n{denoised_proof_str}\n```")
    except Exception as e:
        print(f"Error calling Claude for denoising: {e}")
        return
    print("-" * 30)

    # We stop here for this specific test, no parsing or translation needed.
    print(f"--- Test Case {test_case['name']} Complete ---\n")

async def test_validation_with_override(test_case_name: str, forced_proof_string: str):
    """
    Tests the parsing -> Z3 translation -> Z3 validation pipeline,
    using a forced input proof string (simulating perfect denoising).
    """
    print(f"--- Running Validation Test Case: {test_case_name} ---")
    print("Simulating Denoising Output (Forced Input String):")
    print(f"```\n{forced_proof_string}\n```")
    print("-" * 30)

    # 1. Parse Forced Proof String
    parsed_proof = None
    try:
        parsed_proof = parse_claude_response(forced_proof_string)
        print(f"Parsed Proof: Theorem='{parsed_proof.theorem}', Steps={parsed_proof.steps}")
    except ClaudeResponseParseError as e:
        print(f"Error parsing forced proof string: {e}")
        return # Stop if parsing fails
    print("-" * 30)

    # 2. Build Z3 Translation Prompt
    if not parsed_proof:
        print("Cannot proceed to Z3 translation without a parsed proof.")
        return

    translation_prompt = Z3_TRANSLATION_PROMPT_TEMPLATE.format(
        theorem_statement=parsed_proof.theorem,
        declarations="", # Placeholders for Claude's guidance
        assertion_code=""
    )

    # 3. Call Claude for Z3 Translation
    z3_code_str_cleaned = None
    try:
        print("Calling Claude for Z3 Code Translation...")
        z3_code_str_raw = await call_claude_api(translation_prompt)
        z3_code_str_cleaned = z3_code_str_raw.strip().removeprefix("```python").removesuffix("```").strip()
        print("Received Z3 Code String:")
        print(f"```python\n{z3_code_str_cleaned}\n```")
    except Exception as e:
        print(f"Error calling Claude for Z3 translation: {e}")
        return # Stop if translation fails
    print("-" * 30)

    # 4. Validate with Z3
    if not z3_code_str_cleaned:
        print("Cannot proceed to Z3 validation without Z3 code.")
        return
        
    try:
        print("Calling Z3 Validator...")
        validation_result = validate_proof_with_z3(z3_code_str_cleaned)
        print(f"==> FINAL Validation Result: {validation_result}")
        # You can print details separately if needed:
        # print(f"Validation Details:\n{validation_result.details}")
    except Exception as e:
        print(f"Error during Z3 validation: {e}")
        return
    print("-" * 30)
    print(f"--- Test Case {test_case_name} Complete ---\n")


if __name__ == "__main__":
    # Load environment variables
    print("Loading environment variables...")
    if not find_dotenv():
         print("Warning: .env file not found. Ensure it's in the project root.")
    load_dotenv(find_dotenv())

    if not CLAUDE_API_KEY:
         print("Error: CLAUDE_API_KEY environment variable not set. Cannot run test.")
         sys.exit(1)
         
    # --- Define Target and Test Inputs --- 
    target_clean_tokens = [1011, 1008, 1001, 1009, 1002, 1002, 1003, 1006, 1005, 1001, 1004, 1007, 1012, 1000, 1001, 1002, 1003, 1006, 1005, 1001, 1004, 1007, 1013]
    target_clean_readable = "THEOREM ∀ x ∃ y y = 2 * x + 1 PROOF LET x y = 2 * x + 1 QED"

    test_cases = [
        {
            "name": "Low Noise",
            "noisy_tokens": [1011, 1008, 1001, 1009, 1002, 1002, 1003, 1006, 1005, 1001, 1004, 1007, 1012, 1000, 31517, 1002, 1003, 10309, 1005, 25061, 1004, 1007, 1013],
            "readable_noisy": "THEOREM ∀ x ∃ y y = 2 * x + 1 PROOF LET UNK_31517 y = UNK_10309 * UNK_25061 + 1 QED"
        },
        {
            "name": "Medium Noise",
            "noisy_tokens": [1011, 13702, 7207, 1009, 14162, 4238, 16475, 1006, 1005, 1001, 1004, 46760, 1012, 42874, 1183, 568, 5915, 48261, 45833, 1001, 1004, 1007, 14304],
            "readable_noisy": "THEOREM UNK_13702 UNK_7207 ∃ UNK_14162 UNK_4238 UNK_16475 2 * x + UNK_46760 PROOF UNK_42874 UNK_1183 UNK_568 UNK_5915 UNK_48261 UNK_45833 x + 1 UNK_14304"
        },
        {
            "name": "Medium Noise (Hyper Schedule?)",
            "noisy_tokens": [1011, 1008, 29532, 1009, 1002, 1002, 1003, 1006, 1005, 1001, 1004, 592, 1012, 28069, 49183, 1002, 1003, 36472, 1005, 1001, 36733, 1007, 38344],
            "readable_noisy": "THEOREM ∀ UNK_29532 ∃ y y = 2 * x + UNK_592 PROOF UNK_28069 UNK_49183 y = UNK_36472 * x UNK_36733 1 UNK_38344"
        },
        # Add the fully noisy case as well
        {
            "name": "High Noise (Original Example)",
            "noisy_tokens": [37285, 47074, 40067, 42897, 31763, 1002, 3749, 1006, 20757, 13799, 14886, 18184, 7435, 27235, 35467, 1354, 1003, 41464, 35753, 891, 9988, 20824, 50020],
            "readable_noisy": "UNK_37285 UNK_47074 UNK_40067 UNK_42897 UNK_31763 y UNK_3749 2 UNK_20757 UNK_13799 UNK_14886 UNK_18184 UNK_7435 UNK_27235 UNK_35467 UNK_1354 = UNK_41464 UNK_35753 UNK_891 UNK_9988 UNK_20824 UNK_50020"

        }
    ]

    test_alpha_t = 0.5 # Using a fixed alpha for all tests, adjust if needed

    async def main():
        print(f"Target Clean Tokens: {target_clean_tokens}")
        print(f"Target Clean Readable: {target_clean_readable}\n")
        for case in test_cases:
            await run_denoising_test(case, test_alpha_t, target_clean_readable)
        # Run the test with the forced clean string
        await test_validation_with_override(
            test_case_name="Forced Clean Proof Test",
            forced_proof_string="THEOREM forall x: exists y: y == 2 * x + 1 PROOF LET x y = 2 * x + 1 QED"
        )

    # Run the async main function
    try:
        asyncio.run(main())
    except Exception as e:
         print(f"An error occurred running the test suite: {e}") 