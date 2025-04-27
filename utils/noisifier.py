import random
import string
from typing import Optional

def add_noise_to_text(text: str, noise_level: float = 0.1, seed: Optional[int] = None) -> str:
    """
    Adds noise to a string by randomly substituting characters.

    Args:
        text: The input string.
        noise_level: The probability (0.0 to 1.0) that any given character will be replaced.
        seed: Optional random seed for reproducibility.

    Returns:
        The text with noise added.
    """
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("Noise level must be between 0.0 and 1.0")

    if seed is not None:
        random.seed(seed)

    noisy_chars = []
    # Include basic punctuation and digits along with letters
    possible_chars = string.ascii_letters + string.digits + string.punctuation + ' ' 

    for char in text:
        if random.random() < noise_level:
            # Replace with a random character (could be the same, but less likely)
            noisy_chars.append(random.choice(possible_chars))
        else:
            # Keep the original character
            noisy_chars.append(char)

    return "".join(noisy_chars)

# Example Usage:
if __name__ == '__main__':
    original_text = "Prove that there are infinitely many prime numbers."
    
    print(f"Original: '{original_text}'")

    noisy_text_10 = add_noise_to_text(original_text, noise_level=0.1, seed=42)
    print(f"Noise 0.1: '{noisy_text_10}'")
    
    noisy_text_25 = add_noise_to_text(original_text, noise_level=0.25, seed=42)
    print(f"Noise 0.25: '{noisy_text_25}'")

    noisy_text_50 = add_noise_to_text(original_text, noise_level=0.5, seed=42)
    print(f"Noise 0.5: '{noisy_text_50}'") 