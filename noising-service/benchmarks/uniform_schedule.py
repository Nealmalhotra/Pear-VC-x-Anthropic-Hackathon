import numpy as np
from typing import List, Union, Optional

class UniformNoiseSchedule:
    """
    Uniform noise schedule implementation for comparison with SEDD.
    
    This class implements a simple uniform noise schedule where tokens
    are corrupted with a constant probability at each timestep.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_timesteps: int = 1000,
        beta_min: float = 0.0001,
        beta_max: float = 0.02
    ):
        """
        Initialize the uniform noise schedule.
        
        Args:
            vocab_size: Size of the vocabulary
            num_timesteps: Number of diffusion steps
            beta_min: Minimum noise level (beta) value
            beta_max: Maximum noise level (beta) value
        """
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # Initialize noise schedule parameters
        self.betas = np.linspace(beta_min, beta_max, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
    
    def q_sample(
        self, 
        x_start: np.ndarray, 
        t: Union[int, np.ndarray]
    ) -> np.ndarray:
        """
        Forward diffusion process with uniform noise: q(x_t | x_0)
        
        Args:
            x_start: Initial tokens [batch_size, seq_len]
            t: Timestep(s) to sample at
            
        Returns:
            Noised tokens at timestep t
        """
        if isinstance(t, int):
            t = np.array([t])
                
        # Get alpha_cumprod for this timestep
        alpha_cumprod = self.alphas_cumprod[t]
        while len(alpha_cumprod.shape) < len(x_start.shape):
            alpha_cumprod = np.expand_dims(alpha_cumprod, -1)
        
        # Generate random noise
        noise = np.random.randint(0, self.vocab_size, size=x_start.shape)
        
        # Apply uniform corruption (purely random token selection)
        keep_mask = np.random.random(size=x_start.shape) < alpha_cumprod
        sample = np.where(keep_mask, x_start, noise)
            
        return sample
    
    def get_alpha_for_timestep(self, t: int) -> float:
        """
        Get the alpha (noise level) for a specific timestep.
        
        Args:
            t: The timestep
            
        Returns:
            The alpha value at timestep t
        """
        if t < 0 or t >= self.num_timesteps:
            raise ValueError(f"Timestep {t} is out of range [0, {self.num_timesteps-1}]")
        
        return self.alphas_cumprod[t]


class UniformNoiser:
    """
    Simple uniform noiser for comparing with SEDD TokenNoiser.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_timesteps: int = 1000
    ):
        """
        Initialize the uniform noiser.
        
        Args:
            vocab_size: Size of the vocabulary
            num_timesteps: Number of diffusion steps
        """
        self.vocab_size = vocab_size
        self.noise_schedule = UniformNoiseSchedule(
            vocab_size=vocab_size,
            num_timesteps=num_timesteps
        )
    
    def apply_noise(
        self,
        tokens: List[int],
        timestep: int,
        use_hyperschedule: bool = False,  # Not used, just for interface compatibility
        token_weights: Optional[List[float]] = None  # Not used, just for interface compatibility
    ) -> List[int]:
        """
        Apply uniform noise to a token sequence.
        
        Args:
            tokens: List of token IDs
            timestep: Current diffusion timestep
            use_hyperschedule: Not used, included for interface compatibility
            token_weights: Not used, included for interface compatibility
            
        Returns:
            List of noised tokens
        """
        # Convert to numpy array
        tokens_array = np.array(tokens).reshape(1, -1)  # Add batch dimension
        
        # Apply uniform noise
        noised_tokens = self.noise_schedule.q_sample(tokens_array, timestep)
            
        # Return as list
        return noised_tokens.flatten().tolist()
    
    def get_noise_level(self, timestep: int) -> float:
        """
        Get the noise level (alpha) for a specific timestep.
        
        Args:
            timestep: The diffusion timestep
            
        Returns:
            The alpha value representing how much of the original signal remains
        """
        return self.noise_schedule.get_alpha_for_timestep(timestep)
