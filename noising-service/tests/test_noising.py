import pytest
import numpy as np
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.noise_schedule import SEDDNoiseSchedule, TokenNoiser

# Define token constants to simulate a math proof tokenization
# These values are placeholder IDs - in a real implementation, they would come from the tokenizer
TOKEN_LET = 1000      # Let
TOKEN_VAR_X = 1001    # x variable
TOKEN_VAR_Y = 1002    # y variable
TOKEN_EQUALS = 1003   # =
TOKEN_PLUS = 1004     # +
TOKEN_MULT = 1005     # *
TOKEN_INT_2 = 1006    # 2
TOKEN_INT_1 = 1007    # 1
TOKEN_FORALL = 1008   # ∀
TOKEN_EXISTS = 1009   # ∃
TOKEN_IMPLIES = 1010  # ⇒
TOKEN_THEOREM = 1011  # THEOREM
TOKEN_PROOF = 1012    # PROOF
TOKEN_QED = 1013      # QED
TOKEN_INTEGRAL = 1014 # ∫
TOKEN_PARTIAL = 1015  # ∂

# Sample proof token sequence: "THEOREM: For all x, there exists y such that y = 2x + 1. PROOF: Let x be any real number. Then y = 2x + 1 is a real number. QED"
PROOF_TOKENS = [
    TOKEN_THEOREM, TOKEN_FORALL, TOKEN_VAR_X, TOKEN_EXISTS, TOKEN_VAR_Y, 
    TOKEN_VAR_Y, TOKEN_EQUALS, TOKEN_INT_2, TOKEN_MULT, TOKEN_VAR_X, TOKEN_PLUS, TOKEN_INT_1,
    TOKEN_PROOF, TOKEN_LET, TOKEN_VAR_X, TOKEN_VAR_Y, TOKEN_EQUALS, 
    TOKEN_INT_2, TOKEN_MULT, TOKEN_VAR_X, TOKEN_PLUS, TOKEN_INT_1, TOKEN_QED
]

# Define which tokens are considered critical in a proof (logical operators, keywords)
CRITICAL_TOKENS = {
    TOKEN_EQUALS, TOKEN_PLUS, TOKEN_MULT, TOKEN_FORALL, TOKEN_EXISTS, 
    TOKEN_IMPLIES, TOKEN_THEOREM, TOKEN_PROOF, TOKEN_QED
}

# Define which tokens are less critical (variable names, numbers)
VARIABLE_TOKENS = {TOKEN_VAR_X, TOKEN_VAR_Y, TOKEN_INT_1, TOKEN_INT_2}

# Define weights for math-specific tokens: critical tokens get higher weights (less noise)
def get_math_token_weights(tokens: List[int]) -> List[float]:
    weights = []
    for token in tokens:
        if token in CRITICAL_TOKENS:
            weights.append(2.0)  # Critical tokens: less noise
        elif token in VARIABLE_TOKENS:
            weights.append(0.8)  # Variables and numbers: more noise
        else:
            weights.append(1.0)  # Default weight
    return weights

# Helper function to calculate preservation rate for specific token types
def calculate_preservation_rate(original: List[int], noised: List[int], token_set: set) -> float:
    """Calculate what percentage of tokens from a specific set were preserved."""
    if not original or not noised:
        return 0.0
    
    preserved = 0
    total = 0
    
    for i, token in enumerate(original):
        if token in token_set:
            total += 1
            if i < len(noised) and original[i] == noised[i]:
                preserved += 1
    
    return (preserved / total * 100) if total > 0 else 0.0

class TestSEDDNoiseSchedule:
    """Tests for the SEDD noise schedule implementation."""
    
    def test_initialization(self):
        """Test that the noise schedule initializes correctly."""
        vocab_size = 100
        num_timesteps = 500
        
        schedule = SEDDNoiseSchedule(vocab_size=vocab_size, num_timesteps=num_timesteps)
        
        # Check basic attributes
        assert schedule.vocab_size == vocab_size
        assert schedule.num_timesteps == num_timesteps
        assert len(schedule.betas) == num_timesteps
        assert len(schedule.alphas) == num_timesteps
        assert len(schedule.alphas_cumprod) == num_timesteps
        
        # Check cumulative product values (descending from 1 to near 0)
        assert schedule.alphas_cumprod[0] < 1.0 and schedule.alphas_cumprod[0] > 0.9
        assert schedule.alphas_cumprod[-1] > 0.0 and schedule.alphas_cumprod[-1] < 0.1
        assert np.all(np.diff(schedule.alphas_cumprod) <= 0)  # Should be monotonically decreasing
    
    @pytest.mark.parametrize("schedule_type", ["linear", "cosine", "quadratic"])
    def test_schedule_types(self, schedule_type):
        """Test different schedule types."""
        vocab_size = 100
        num_timesteps = 500
        
        schedule = SEDDNoiseSchedule(
            vocab_size=vocab_size, 
            num_timesteps=num_timesteps,
            schedule_type=schedule_type
        )
        
        # All schedule types should have valid beta values
        assert np.all(schedule.betas >= 0)
        assert np.all(schedule.betas <= 1)
        assert len(schedule.betas) == num_timesteps
    
    def test_q_sample_numpy(self):
        """Test forward diffusion process with numpy arrays."""
        vocab_size = 5000  # Increased to accommodate math tokens
        num_timesteps = 500
        batch_size = 2
        
        # Use math proof tokens instead of random values
        x_start = np.array([PROOF_TOKENS] * batch_size)
        seq_len = len(PROOF_TOKENS)
        
        schedule = SEDDNoiseSchedule(vocab_size=vocab_size, num_timesteps=num_timesteps)
        
        # Test at different timesteps
        for t in [0, 100, 400, 499]:
            # Apply noise
            x_t = schedule.q_sample(x_start, t)
            
            # Check shape
            assert x_t.shape == x_start.shape
            
            # Check values are valid tokens
            assert np.all(x_t >= 0)
            assert np.all(x_t < vocab_size)
            
            # Early timesteps should have more original tokens preserved
            if t < 100:
                # Count matching tokens
                matches = np.sum(x_t == x_start)
                total = batch_size * seq_len
                assert matches / total > 0.5  # More than half preserved for early timesteps
    
    def test_get_alpha_for_timestep(self):
        """Test retrieving alpha values for specific timesteps."""
        vocab_size = 5000
        num_timesteps = 500
        
        schedule = SEDDNoiseSchedule(vocab_size=vocab_size, num_timesteps=num_timesteps)
        
        # Test valid timesteps
        for t in [0, 100, 499]:
            alpha = schedule.get_alpha_for_timestep(t)
            assert 0 <= alpha <= 1
        
        # Test invalid timesteps
        with pytest.raises(ValueError):
            schedule.get_alpha_for_timestep(-1)
        
        with pytest.raises(ValueError):
            schedule.get_alpha_for_timestep(500)
    
    def test_apply_hyperschedule_for_math_proof(self):
        """Test applying position-dependent noise with math-proof hyperschedule."""
        vocab_size = 5000
        num_timesteps = 500
        batch_size = 2
        
        # Use proof tokens for test
        x_start = np.array([PROOF_TOKENS] * batch_size)
        
        schedule = SEDDNoiseSchedule(vocab_size=vocab_size, num_timesteps=num_timesteps)
        
        # Create math-specific token weights
        token_weights = np.array([get_math_token_weights(PROOF_TOKENS)] * batch_size)
        
        # Apply hyperschedule noise at moderate timestep
        t = 250
        x_t = schedule.apply_hyperschedule(x_start, t, token_weights)
        
        # Check results have correct shape and values
        assert x_t.shape == x_start.shape
        assert np.all(x_t >= 0)
        assert np.all(x_t < vocab_size)
        
        # Count preserved tokens by type
        preserved_critical = 0
        preserved_variables = 0
        total_critical = 0
        total_variables = 0
        
        for batch_idx in range(batch_size):
            for i, token in enumerate(x_start[batch_idx]):
                if token in CRITICAL_TOKENS:
                    total_critical += 1
                    if x_t[batch_idx, i] == token:
                        preserved_critical += 1
                elif token in VARIABLE_TOKENS:
                    total_variables += 1
                    if x_t[batch_idx, i] == token:
                        preserved_variables += 1
        
        # Calculate preservation rates
        critical_preservation_rate = (preserved_critical / total_critical * 100) if total_critical > 0 else 0
        variable_preservation_rate = (preserved_variables / total_variables * 100) if total_variables > 0 else 0
        
        # Critical tokens should be preserved more often than variable tokens
        print(f"Critical preservation rate: {critical_preservation_rate}%")
        print(f"Variable preservation rate: {variable_preservation_rate}%")
        
        # This might occasionally fail due to randomness, but should be true most of the time
        # If it fails too often, might need to adjust the test or run multiple times
        assert critical_preservation_rate > variable_preservation_rate, \
            f"Critical tokens ({critical_preservation_rate}%) should be preserved more than variables ({variable_preservation_rate}%)"
        
    def test_apply_hyperschedule_with_special_math_tokens(self):
        """Test hyperschedule with special math tokens like integrals and partial derivatives."""
        vocab_size = 5000
        num_timesteps = 500
        
        schedule = SEDDNoiseSchedule(vocab_size=vocab_size, num_timesteps=num_timesteps)
        
        # Special math expression: "∫ ∂f/∂x dx"
        # Using placeholder tokens for special math symbols
        special_math_tokens = np.array([[TOKEN_INTEGRAL, TOKEN_PARTIAL, TOKEN_PARTIAL]], dtype=np.int32)
        
        # Create weights that highly preserve special symbols
        token_weights = np.array([[2.5, 2.5, 2.5]])  # Very high weights for special symbols
        
        # Apply hyperschedule noise
        t = 300  # Significant noise
        special_noised = schedule.apply_hyperschedule(special_math_tokens, t, token_weights)
        
        # Special tokens should mostly be preserved due to high weights
        matches = np.sum(special_noised == special_math_tokens)
        assert matches > 0, "At least some special math tokens should be preserved"


class TestTokenNoiser:
    """Tests for the TokenNoiser class."""
    
    def test_apply_noise_to_math_proof(self):
        """Test applying noise to mathematical proof token sequences."""
        vocab_size = 5000
        num_timesteps = 500
        
        noiser = TokenNoiser(vocab_size=vocab_size, num_timesteps=num_timesteps)
        
        # Test applying noise at different timesteps with math tokens
        for t in [0, 100, 400, 499]:
            noised_tokens = noiser.apply_noise(PROOF_TOKENS, t)
            
            # Check result is a list with correct length
            assert isinstance(noised_tokens, list)
            assert len(noised_tokens) == len(PROOF_TOKENS)
            
            # Check values are valid tokens
            assert all(0 <= token < vocab_size for token in noised_tokens)
            
            # Calculate preservation metrics
            critical_preservation = calculate_preservation_rate(
                PROOF_TOKENS, noised_tokens, CRITICAL_TOKENS
            )
            variable_preservation = calculate_preservation_rate(
                PROOF_TOKENS, noised_tokens, VARIABLE_TOKENS
            )
            
            # At very low timesteps (t=0), almost everything should be preserved
            if t == 0:
                assert critical_preservation > 90, f"At timestep 0, critical tokens should be highly preserved (got {critical_preservation}%)"
                assert variable_preservation > 90, f"At timestep 0, variable tokens should be highly preserved (got {variable_preservation}%)"
            
            # At very high timesteps (t=499), very little should be preserved
            if t == 499:
                assert critical_preservation < 50, f"At high timestep, preservation should be low (got {critical_preservation}%)"
    
    def test_apply_math_proof_hyperschedule(self):
        """Test applying hyperschedule to math proof with weighted token importance."""
        vocab_size = 5000
        num_timesteps = 500
        
        noiser = TokenNoiser(vocab_size=vocab_size, num_timesteps=num_timesteps)
        
        # Get math-specific token weights
        token_weights = get_math_token_weights(PROOF_TOKENS)
        
        # Apply noise with hyperschedule at moderate timestep
        t = 250
        noised_tokens = noiser.apply_noise(
            PROOF_TOKENS, t, use_hyperschedule=True, token_weights=token_weights
        )
        
        # Check result is a list with correct length
        assert isinstance(noised_tokens, list)
        assert len(noised_tokens) == len(PROOF_TOKENS)
        
        # Check values are valid tokens
        assert all(0 <= token < vocab_size for token in noised_tokens)
        
        # Calculate preservation metrics
        critical_preservation = calculate_preservation_rate(
            PROOF_TOKENS, noised_tokens, CRITICAL_TOKENS
        )
        variable_preservation = calculate_preservation_rate(
            PROOF_TOKENS, noised_tokens, VARIABLE_TOKENS
        )
        
        print(f"With hyperschedule - Critical preservation: {critical_preservation}%")
        print(f"With hyperschedule - Variable preservation: {variable_preservation}%")
        
        # Critical tokens (with higher weights) should be preserved more than variables
        assert critical_preservation > variable_preservation, \
            f"Critical tokens ({critical_preservation}%) should be preserved more than variables ({variable_preservation}%)"
        
        # Now test without hyperschedule for comparison
        regular_noised = noiser.apply_noise(
            PROOF_TOKENS, t, use_hyperschedule=False
        )
        
        regular_critical = calculate_preservation_rate(
            PROOF_TOKENS, regular_noised, CRITICAL_TOKENS
        )
        
        # With hyperschedule, critical token preservation should generally be higher 
        # than without it (though this could occasionally fail due to randomness)
        print(f"Without hyperschedule - Critical preservation: {regular_critical}%")
        
        # We don't assert here because this is probabilistic and could occasionally fail
        # but we print the values for inspection
    
    def test_get_noise_level(self):
        """Test retrieving noise levels."""
        vocab_size = 5000
        num_timesteps = 500
        
        noiser = TokenNoiser(vocab_size=vocab_size, num_timesteps=num_timesteps)
        
        # Test valid timesteps
        for t in [0, 100, 499]:
            alpha = noiser.get_noise_level(t)
            assert 0 <= alpha <= 1
        
        # Test invalid timesteps should raise ValueError
        with pytest.raises(ValueError):
            noiser.get_noise_level(-1)
        
        with pytest.raises(ValueError):
            noiser.get_noise_level(500)
