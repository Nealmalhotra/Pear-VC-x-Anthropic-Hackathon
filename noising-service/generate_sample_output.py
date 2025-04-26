import requests
import json
import sys
import os
from typing import Dict, Any, List

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import token constants and proof tokens from test file
from tests.test_noising import (
    PROOF_TOKENS, CRITICAL_TOKENS, VARIABLE_TOKENS, 
    get_math_token_weights, calculate_preservation_rate,
    TOKEN_THEOREM, TOKEN_FORALL, TOKEN_VAR_X, TOKEN_EXISTS, TOKEN_VAR_Y,
    TOKEN_EQUALS, TOKEN_INT_2, TOKEN_MULT, TOKEN_PLUS, TOKEN_INT_1,
    TOKEN_PROOF, TOKEN_LET, TOKEN_QED
)

def token_id_to_name(token_id: int) -> str:
    """Convert token ID to a human-readable name for demonstration purposes."""
    token_names = {
        TOKEN_THEOREM: "THEOREM",
        TOKEN_FORALL: "∀",
        TOKEN_VAR_X: "x",
        TOKEN_EXISTS: "∃",
        TOKEN_VAR_Y: "y",
        TOKEN_EQUALS: "=",
        TOKEN_INT_2: "2",
        TOKEN_MULT: "*",
        TOKEN_PLUS: "+",
        TOKEN_INT_1: "1",
        TOKEN_PROOF: "PROOF",
        TOKEN_LET: "LET",
        TOKEN_QED: "QED"
    }
    return token_names.get(token_id, f"UNK_{token_id}")

def tokens_to_readable(tokens: List[int]) -> str:
    """Convert token IDs to readable text."""
    return " ".join(token_id_to_name(token) for token in tokens)

def generate_sample_output(server_url: str = "http://localhost:8000"):
    """Generate sample output from the noising service at different timesteps."""
    # The original proof tokens
    print(f"Original Proof Tokens: {PROOF_TOKENS}")
    print(f"As Text: {tokens_to_readable(PROOF_TOKENS)}")
    print("\n" + "="*80 + "\n")
    
    # Sample at different noise levels (timesteps)
    timesteps = [0, 100, 250, 500, 750, 999]
    
    results = {}
    
    for timestep in timesteps:
        # 1. Standard noising (no hyperschedule)
        request_data = {
            "tokens": PROOF_TOKENS,
            "timestep": timestep
        }
        
        try:
            # For local testing, use direct API import instead of HTTP request
            from src.api import NoiseRequest, get_token_noiser, apply_noise
            from fastapi.testclient import TestClient
            from src.api import app
            
            client = TestClient(app)
            response = client.post("/noise", json=request_data)
            
            if response.status_code == 200:
                data = response.json()
                
                # Calculate preservation metrics
                noised_tokens = data["noised_tokens"]
                alpha = data["alpha"]
                critical_rate = calculate_preservation_rate(
                    PROOF_TOKENS, noised_tokens, CRITICAL_TOKENS
                )
                variable_rate = calculate_preservation_rate(
                    PROOF_TOKENS, noised_tokens, VARIABLE_TOKENS
                )
                
                print(f"TIMESTEP {timestep} (Alpha: {alpha:.4f}):")
                print(f"  Noised tokens: {noised_tokens}")
                print(f"  As Text: {tokens_to_readable(noised_tokens)}")
                print(f"  Preservation: {data['metrics']['percent_unchanged']:.2f}% overall")
                print(f"  Critical tokens preserved: {critical_rate:.2f}%")
                print(f"  Variable tokens preserved: {variable_rate:.2f}%")
                
                # Save for final output
                results[timestep] = {
                    "standard": {
                        "noised_tokens": noised_tokens,
                        "alpha": alpha,
                        "preservation_rate": data['metrics']['percent_unchanged'],
                        "critical_preservation": critical_rate,
                        "variable_preservation": variable_rate
                    }
                }
                
                # 2. With hyperschedule for mathematical proofs
                if timestep in [250, 500]:  # Only do hyperschedule for midrange timesteps
                    token_weights = get_math_token_weights(PROOF_TOKENS)
                    hyper_request = {
                        "tokens": PROOF_TOKENS,
                        "timestep": timestep,
                        "use_hyperschedule": True,
                        "token_weights": token_weights
                    }
                    
                    hyper_response = client.post("/noise", json=hyper_request)
                    if hyper_response.status_code == 200:
                        hyper_data = hyper_response.json()
                        hyper_noised = hyper_data["noised_tokens"]
                        hyper_critical = calculate_preservation_rate(
                            PROOF_TOKENS, hyper_noised, CRITICAL_TOKENS
                        )
                        hyper_variable = calculate_preservation_rate(
                            PROOF_TOKENS, hyper_noised, VARIABLE_TOKENS
                        )
                        
                        print(f"\n  With Hyperschedule:")
                        print(f"  Noised tokens: {hyper_noised}")
                        print(f"  As Text: {tokens_to_readable(hyper_noised)}")
                        print(f"  Preservation: {hyper_data['metrics']['percent_unchanged']:.2f}% overall")
                        print(f"  Critical tokens preserved: {hyper_critical:.2f}%")
                        print(f"  Variable tokens preserved: {hyper_variable:.2f}%")
                        
                        # Save hyperschedule results
                        results[timestep]["hyperschedule"] = {
                            "noised_tokens": hyper_noised,
                            "alpha": alpha,  # Same base alpha
                            "preservation_rate": hyper_data['metrics']['percent_unchanged'],
                            "critical_preservation": hyper_critical,
                            "variable_preservation": hyper_variable
                        }
                
                print("\n" + "-"*50 + "\n")
            else:
                print(f"Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error generating noised output: {str(e)}")
    
    # Generate final JSON output for the denoiser service
    print("\n" + "="*80)
    print("SAMPLE JSON FOR DENOISER SERVICE:")
    print("="*80 + "\n")
    
    # Select a few representative examples
    denoiser_examples = {
        "original_tokens": PROOF_TOKENS,
        "original_readable": tokens_to_readable(PROOF_TOKENS),
        "examples": [
            # Low noise example
            {
                "timestep": 100,
                "noised_tokens": results[100]["standard"]["noised_tokens"],
                "alpha": results[100]["standard"]["alpha"],
                "readable": tokens_to_readable(results[100]["standard"]["noised_tokens"])
            },
            # Medium noise example
            {
                "timestep": 500,
                "noised_tokens": results[500]["standard"]["noised_tokens"],
                "alpha": results[500]["standard"]["alpha"],
                "readable": tokens_to_readable(results[500]["standard"]["noised_tokens"])
            },
            # Medium noise with hyperschedule
            {
                "timestep": 500,
                "noised_tokens": results[500]["hyperschedule"]["noised_tokens"],
                "alpha": results[500]["hyperschedule"]["alpha"],
                "use_hyperschedule": True,
                "token_weights": get_math_token_weights(PROOF_TOKENS),
                "readable": tokens_to_readable(results[500]["hyperschedule"]["noised_tokens"])
            },
            # High noise example
            {
                "timestep": 750,
                "noised_tokens": results[750]["standard"]["noised_tokens"],
                "alpha": results[750]["standard"]["alpha"],
                "readable": tokens_to_readable(results[750]["standard"]["noised_tokens"])
            }
        ]
    }
    
    print(json.dumps(denoiser_examples, indent=2))
    
    # Save to file for easy sharing
    with open('denoiser_input_examples.json', 'w') as f:
        json.dump(denoiser_examples, f, indent=2)
    
    print(f"\nSaved examples to denoiser_input_examples.json")

if __name__ == "__main__":
    generate_sample_output() 