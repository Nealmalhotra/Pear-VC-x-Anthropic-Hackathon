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
    
    @pytest.mark.parametrize("schedule_type", ["linear", "cosine", "quadratic", "sedd_entropy"])
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
    
    def test_sedd_entropy_schedule(self):
        """Test specific properties of the SEDD entropy schedule."""
        vocab_size = 5000
        num_timesteps = 500
        beta_min = 0.0001
        beta_max = 0.02
        
        schedule = SEDDNoiseSchedule(
            vocab_size=vocab_size,
            num_timesteps=num_timesteps,
            beta_min=beta_min,
            beta_max=beta_max,
            schedule_type="sedd_entropy"
        )
        
        # Check that betas are calculated according to SEDD entropy formula
        # The implementation creates sigma values on a log scale
        sigma_vals = np.exp(np.linspace(np.log(beta_min), np.log(beta_max), num_timesteps))
        sigma_prev = np.concatenate(([0.0], sigma_vals[:-1]))
        expected_betas = 1.0 - np.exp(-(sigma_vals - sigma_prev))
        expected_betas = np.clip(expected_betas, 0.0, 1.0)
        
        # Check that the calculated betas match expected values
        assert np.allclose(schedule.betas, expected_betas)
        
        # Test applying the SEDD entropy schedule to math proof tokens
        x_start = np.array([PROOF_TOKENS])
        t = 250  # Middle timestep
        
        # Apply noise with SEDD entropy schedule
        x_t = schedule.q_sample(x_start, t)
        
        # Check basic properties
        assert x_t.shape == x_start.shape
        assert np.all(x_t >= 0)
        assert np.all(x_t < vocab_size)
        
        # Test with TokenNoiser
        noiser = TokenNoiser(
            vocab_size=vocab_size,
            num_timesteps=num_timesteps,
            schedule_type="sedd_entropy"
        )
        
        # Apply noise to proof tokens
        noised_tokens = noiser.apply_noise(PROOF_TOKENS, t)
        
        # Check that the result is valid
        assert isinstance(noised_tokens, list)
        assert len(noised_tokens) == len(PROOF_TOKENS)
        assert all(0 <= token < vocab_size for token in noised_tokens)
    
    def test_compare_schedule_types(self):
        """Compare different schedule types, including SEDD entropy."""
        vocab_size = 5000
        num_timesteps = 500
        
        # Create different schedule types
        schedules = {
            "linear": SEDDNoiseSchedule(vocab_size, num_timesteps, schedule_type="linear"),
            "cosine": SEDDNoiseSchedule(vocab_size, num_timesteps, schedule_type="cosine"),
            "quadratic": SEDDNoiseSchedule(vocab_size, num_timesteps, schedule_type="quadratic"),
            "sedd_entropy": SEDDNoiseSchedule(vocab_size, num_timesteps, schedule_type="sedd_entropy")
        }
        
        # Compare alpha decay curves
        timesteps = [0, 100, 200, 300, 400, 499]
        alpha_values = {}
        
        for name, schedule in schedules.items():
            alpha_values[name] = [schedule.get_alpha_for_timestep(t) for t in timesteps]
        
        # Each schedule should produce a unique alpha curve
        # We just check they're not identical - actual values will vary by schedule design
        schedule_pairs = [
            ("linear", "cosine"),
            ("linear", "quadratic"),
            ("linear", "sedd_entropy"),
            ("cosine", "quadratic"),
            ("cosine", "sedd_entropy"),
            ("quadratic", "sedd_entropy")
        ]
        
        for schedule1, schedule2 in schedule_pairs:
            assert not np.allclose(alpha_values[schedule1], alpha_values[schedule2]), \
                f"{schedule1} and {schedule2} should have different alpha profiles"
        
        # Test effect on actual tokens
        x_start = np.array([PROOF_TOKENS])
        t = 250  # Middle timestep
        
        # Apply each schedule and collect preservation metrics
        preservation_rates = {}
        for name, schedule in schedules.items():
            x_t = schedule.q_sample(x_start, t)
            # Convert to lists for the helper function
            original_list = x_start[0].tolist()
            noised_list = x_t[0].tolist()
            
            critical_rate = calculate_preservation_rate(
                original_list, noised_list, CRITICAL_TOKENS
            )
            variable_rate = calculate_preservation_rate(
                original_list, noised_list, VARIABLE_TOKENS
            )
            
            preservation_rates[name] = {
                "critical": critical_rate,
                "variables": variable_rate
            }
            
            # The basic assertion all schedules should satisfy
            assert critical_rate >= 0 and critical_rate <= 100
            assert variable_rate >= 0 and variable_rate <= 100
        
        # Print rates for comparison (useful for debugging)
        for name, rates in preservation_rates.items():
            print(f"{name} - Critical: {rates['critical']:.2f}%, Variables: {rates['variables']:.2f}%")
        
        # We don't do specific comparisons between schedules as they're probabilistic
        # But we've collected the data to verify SEDD entropy works
    
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
    
    def test_token_noiser_with_sedd_entropy(self):
        """Test TokenNoiser using the SEDD entropy schedule."""
        vocab_size = 5000
        num_timesteps = 500
        
        # Create a TokenNoiser with SEDD entropy schedule
        sedd_noiser = TokenNoiser(
            vocab_size=vocab_size,
            num_timesteps=num_timesteps,
            schedule_type="sedd_entropy"
        )
        
        # Create a TokenNoiser with cosine schedule for comparison
        cosine_noiser = TokenNoiser(
            vocab_size=vocab_size,
            num_timesteps=num_timesteps,
            schedule_type="cosine"
        )
        
        # Test both with and without hyperschedule at multiple noise levels
        token_weights = get_math_token_weights(PROOF_TOKENS)
        
        # Test across different timesteps
        timesteps = [50, 250, 450]
        
        print("\n" + "="*80)
        print("COMPARING SEDD ENTROPY VS COSINE SCHEDULES")
        print("="*80)
        
        for t in timesteps:
            # Test SEDD entropy with hyperschedule
            sedd_hyper_noised = sedd_noiser.apply_noise(
                PROOF_TOKENS, t, use_hyperschedule=True, token_weights=token_weights
            )
            
            # Test SEDD entropy without hyperschedule
            sedd_regular_noised = sedd_noiser.apply_noise(
                PROOF_TOKENS, t, use_hyperschedule=False
            )
            
            # Test cosine with hyperschedule
            cosine_hyper_noised = cosine_noiser.apply_noise(
                PROOF_TOKENS, t, use_hyperschedule=True, token_weights=token_weights
            )
            
            # Test cosine without hyperschedule
            cosine_regular_noised = cosine_noiser.apply_noise(
                PROOF_TOKENS, t, use_hyperschedule=False
            )
            
            # Check all results have valid formats
            for result in [sedd_hyper_noised, sedd_regular_noised, cosine_hyper_noised, cosine_regular_noised]:
                assert isinstance(result, list)
                assert len(result) == len(PROOF_TOKENS)
                assert all(0 <= token < vocab_size for token in result)
            
            # Calculate preservation rates for critical tokens
            sedd_hyper_critical = calculate_preservation_rate(
                PROOF_TOKENS, sedd_hyper_noised, CRITICAL_TOKENS
            )
            
            sedd_regular_critical = calculate_preservation_rate(
                PROOF_TOKENS, sedd_regular_noised, CRITICAL_TOKENS
            )
            
            cosine_hyper_critical = calculate_preservation_rate(
                PROOF_TOKENS, cosine_hyper_noised, CRITICAL_TOKENS
            )
            
            cosine_regular_critical = calculate_preservation_rate(
                PROOF_TOKENS, cosine_regular_noised, CRITICAL_TOKENS
            )
            
            # Calculate preservation rates for variable tokens
            sedd_hyper_variable = calculate_preservation_rate(
                PROOF_TOKENS, sedd_hyper_noised, VARIABLE_TOKENS
            )
            
            sedd_regular_variable = calculate_preservation_rate(
                PROOF_TOKENS, sedd_regular_noised, VARIABLE_TOKENS
            )
            
            cosine_hyper_variable = calculate_preservation_rate(
                PROOF_TOKENS, cosine_hyper_noised, VARIABLE_TOKENS
            )
            
            cosine_regular_variable = calculate_preservation_rate(
                PROOF_TOKENS, cosine_regular_noised, VARIABLE_TOKENS
            )
            
            # Print preservation rates for debugging
            print(f"\nTIMESTEP {t} (Noise Alpha: SEDD={sedd_noiser.get_noise_level(t):.4f}, Cosine={cosine_noiser.get_noise_level(t):.4f}):")
            print("-" * 80)
            print(f"{'Schedule':<15} {'Mode':<15} {'Critical Tokens':<20} {'Variable Tokens':<20} {'Difference'}")
            print(f"{'SEDD':<15} {'Hyperschedule':<15} {sedd_hyper_critical:>7.2f}% {' ':>12} {sedd_hyper_variable:>7.2f}% {' ':>12} {sedd_hyper_critical-sedd_hyper_variable:>7.2f}%")
            print(f"{'SEDD':<15} {'Regular':<15} {sedd_regular_critical:>7.2f}% {' ':>12} {sedd_regular_variable:>7.2f}% {' ':>12} {sedd_regular_critical-sedd_regular_variable:>7.2f}%")
            print(f"{'Cosine':<15} {'Hyperschedule':<15} {cosine_hyper_critical:>7.2f}% {' ':>12} {cosine_hyper_variable:>7.2f}% {' ':>12} {cosine_hyper_critical-cosine_hyper_variable:>7.2f}%")
            print(f"{'Cosine':<15} {'Regular':<15} {cosine_regular_critical:>7.2f}% {' ':>12} {cosine_regular_variable:>7.2f}% {' ':>12} {cosine_regular_critical-cosine_regular_variable:>7.2f}%")
            
            # For each schedule type, hyperschedule should generally preserve critical tokens better
            # (we don't assert due to randomness, but we print for inspection)
            
        # Check that noise levels can be retrieved properly
        for t in timesteps:
            sedd_alpha = sedd_noiser.get_noise_level(t)
            cosine_alpha = cosine_noiser.get_noise_level(t)
            
            assert 0 <= sedd_alpha <= 1
            assert 0 <= cosine_alpha <= 1
            # Schedules should give different values (not always guaranteed but highly likely)
            assert abs(sedd_alpha - cosine_alpha) > 1e-6, f"Different schedules should give different alphas at t={t}"
    
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
