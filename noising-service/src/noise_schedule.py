import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional, Union

class SEDDNoiseSchedule:
    """
    Score Entropy Discrete Diffusion (SEDD) noise schedule implementation.
    
    This class implements the entropy-based schedule from the SEDD paper
    for generating corrupt tokens at each diffusion timestep.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_timesteps: int = 1000,
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        schedule_type: str = "cosine"
    ):
        """
        Initialize the SEDD noise schedule.
        
        Args:
            vocab_size: Size of the vocabulary
            num_timesteps: Number of diffusion steps
            beta_min: Minimum noise level (beta) value
            beta_max: Maximum noise level (beta) value
            schedule_type: Type of schedule ('linear', 'cosine', or 'quadratic')
        """
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.schedule_type = schedule_type
        
        # Initialize noise schedule parameters
        self.betas = self._create_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # Pre-compute values for the forward process
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _create_noise_schedule(self) -> np.ndarray:
        """
        Create the noise schedule based on the specified schedule type.
        
        Returns:
            np.ndarray: The beta (noise level) schedule
        """
        if self.schedule_type == "linear":
            return np.linspace(self.beta_min, self.beta_max, self.num_timesteps)
        
        elif self.schedule_type == "cosine":
            # Implementation of improved cosine schedule as per DDPM/SEDD papers
            steps = self.num_timesteps + 1
            x = np.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = np.cos(((x / self.num_timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0.0, 0.999)
        
        elif self.schedule_type == "quadratic":
            return np.linspace(self.beta_min**0.5, self.beta_max**0.5, self.num_timesteps) ** 2
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def q_sample(
        self, 
        x_start: Union[np.ndarray, torch.Tensor], 
        t: Union[int, np.ndarray, torch.Tensor], 
        noise: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_start: Initial tokens [batch_size, seq_len]
            t: Timestep(s) to sample at
            noise: Optional pre-generated noise
            
        Returns:
            Noised tokens at timestep t
        """
        is_tensor = isinstance(x_start, torch.Tensor)
        device = x_start.device if is_tensor else None
        
        if isinstance(t, int):
            t = np.array([t])
            if is_tensor:
                t = torch.from_numpy(t).to(device)
                
        # Get alpha_cumprod for this timestep
        if is_tensor:
            alpha_cumprod = torch.tensor(self.alphas_cumprod, device=device)[t]
            while len(alpha_cumprod.shape) < len(x_start.shape):
                alpha_cumprod = alpha_cumprod.unsqueeze(-1)
        else:
            alpha_cumprod = self.alphas_cumprod[t]
            while len(alpha_cumprod.shape) < len(x_start.shape):
                alpha_cumprod = np.expand_dims(alpha_cumprod, -1)
        
        # Generate random noise if not provided
        if noise is None:
            if is_tensor:
                noise = torch.randint(0, self.vocab_size, x_start.shape, device=device)
            else:
                noise = np.random.randint(0, self.vocab_size, size=x_start.shape)
        
        # Apply SEDD's entropy-based corruption (choosing between original and noise)
        if is_tensor:
            keep_mask = torch.rand_like(x_start.float()) < alpha_cumprod
            sample = torch.where(keep_mask, x_start, noise)
        else:
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
    
    def apply_hyperschedule(
        self, 
        x_start: Union[np.ndarray, torch.Tensor], 
        t: int, 
        token_weights: Union[np.ndarray, torch.Tensor, None] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply a position-dependent hyperschedule that varies noise per token.
        
        Args:
            x_start: Initial tokens [batch_size, seq_len]
            t: Timestep to sample at
            token_weights: Optional weights for each token position [seq_len]
                           Higher values = less noise applied
                           
        Returns:
            Noised tokens with position-dependent noise levels
        """
        is_tensor = isinstance(x_start, torch.Tensor)
        device = x_start.device if is_tensor else None
        batch_size, seq_len = x_start.shape[0], x_start.shape[1]
        
        # Default weights if none provided
        if token_weights is None:
            if is_tensor:
                token_weights = torch.ones(seq_len, device=device)
            else:
                token_weights = np.ones(seq_len)
        
        # Ensure weights are properly shaped and normalized
        if is_tensor:
            token_weights = token_weights.to(device)
            if len(token_weights.shape) == 1:
                token_weights = token_weights.unsqueeze(0).repeat(batch_size, 1)
            # Normalize to [0.5, 1.5] range to adjust noise levels
            token_weights = 0.5 + token_weights / token_weights.max()
        else:
            if len(token_weights.shape) == 1:
                token_weights = np.tile(token_weights[np.newaxis, :], (batch_size, 1))
            # Normalize to [0.5, 1.5] range to adjust noise levels
            token_weights = 0.5 + token_weights / token_weights.max()
            
        # Get base alpha for this timestep
        base_alpha = self.alphas_cumprod[t]
        
        # Adjust alpha per token position
        if is_tensor:
            adjusted_alphas = torch.clamp(base_alpha * token_weights, 0.001, 0.999)
            # Generate noise
            noise = torch.randint(0, self.vocab_size, x_start.shape, device=device)
            # Apply position-dependent noise
            keep_mask = torch.rand_like(x_start.float()) < adjusted_alphas
            sample = torch.where(keep_mask, x_start, noise)
        else:
            adjusted_alphas = np.clip(base_alpha * token_weights, 0.001, 0.999)
            # Generate noise
            noise = np.random.randint(0, self.vocab_size, size=x_start.shape)
            # Apply position-dependent noise
            keep_mask = np.random.random(size=x_start.shape) < adjusted_alphas
            sample = np.where(keep_mask, x_start, noise)
            
        return sample


class TokenNoiser:
    """
    Main class to apply noise to token sequences according to SEDD principles.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine"
    ):
        """
        Initialize the token noiser.
        
        Args:
            vocab_size: Size of the vocabulary
            num_timesteps: Number of diffusion steps
            schedule_type: Type of noise schedule
        """
        self.vocab_size = vocab_size
        self.noise_schedule = SEDDNoiseSchedule(
            vocab_size=vocab_size,
            num_timesteps=num_timesteps,
            schedule_type=schedule_type
        )
    
    def apply_noise(
        self,
        tokens: List[int],
        timestep: int,
        use_hyperschedule: bool = False,
        token_weights: Optional[List[float]] = None
    ) -> List[int]:
        """
        Apply noise to a token sequence according to the noise schedule.
        
        Args:
            tokens: List of token IDs
            timestep: Current diffusion timestep
            use_hyperschedule: Whether to use position-dependent noise 
            token_weights: Optional weights for each token position
            
        Returns:
            List of noised tokens
        """
        # Convert to numpy array
        tokens_array = np.array(tokens).reshape(1, -1)  # Add batch dimension
        
        # Apply noise
        if use_hyperschedule and token_weights is not None:
            weights_array = np.array(token_weights)
            noised_tokens = self.noise_schedule.apply_hyperschedule(
                tokens_array, timestep, weights_array
            )
        else:
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
