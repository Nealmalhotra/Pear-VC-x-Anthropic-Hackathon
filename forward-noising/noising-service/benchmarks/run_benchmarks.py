import time
import numpy as np
import sys
import os
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import json

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.noise_schedule import SEDDNoiseSchedule, TokenNoiser
from benchmarks.uniform_schedule import UniformNoiseSchedule

def generate_sample_tokens(length: int, vocab_size: int) -> List[int]:
    """Generate a random sequence of tokens for testing."""
    return np.random.randint(0, vocab_size, size=length).tolist()

def benchmark_noising(
    tokens: List[int], 
    timesteps: List[int], 
    noiser_sedd: TokenNoiser, 
    noiser_uniform: 'UniformNoiser',
    runs: int = 5
) -> Dict[str, Any]:
    """
    Benchmark SEDD vs Uniform noising for different timesteps.
    
    Args:
        tokens: List of token IDs
        timesteps: List of timesteps to benchmark
        noiser_sedd: SEDD TokenNoiser instance
        noiser_uniform: Uniform noiser instance
        runs: Number of runs for averaging
        
    Returns:
        Benchmark results dictionary
    """
    results = {
        "tokens_length": len(tokens),
        "vocab_size": noiser_sedd.vocab_size,
        "timesteps": timesteps,
        "sedd_times": [],
        "uniform_times": [],
        "sedd_preservation": [],
        "uniform_preservation": []
    }
    
    # Run benchmarks for each timestep
    for t in timesteps:
        # Benchmark SEDD
        sedd_time = 0
        sedd_preserved = 0
        for _ in range(runs):
            start = time.time()
            noised = noiser_sedd.apply_noise(tokens, t)
            sedd_time += (time.time() - start)
            # Count preserved tokens
            preserved = sum(1 for i, token in enumerate(noised) if token == tokens[i])
            sedd_preserved += preserved / len(tokens)
        
        # Average over runs
        results["sedd_times"].append(sedd_time / runs)
        results["sedd_preservation"].append(sedd_preserved / runs * 100)  # as percentage
        
        # Benchmark Uniform
        uniform_time = 0
        uniform_preserved = 0
        for _ in range(runs):
            start = time.time()
            noised = noiser_uniform.apply_noise(tokens, t)
            uniform_time += (time.time() - start)
            # Count preserved tokens
            preserved = sum(1 for i, token in enumerate(noised) if token == tokens[i])
            uniform_preserved += preserved / len(tokens)
        
        # Average over runs
        results["uniform_times"].append(uniform_time / runs)
        results["uniform_preservation"].append(uniform_preserved / runs * 100)  # as percentage
    
    return results

def plot_results(results: Dict[str, Any], output_file: str = "benchmark_results.png"):
    """Plot benchmark results and save to file."""
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Time comparison
    plt.subplot(2, 1, 1)
    plt.plot(results["timesteps"], results["sedd_times"], 'b-', label='SEDD')
    plt.plot(results["timesteps"], results["uniform_times"], 'r-', label='Uniform')
    plt.xlabel('Timestep')
    plt.ylabel('Time (s)')
    plt.title('Noising Performance: SEDD vs Uniform')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Token preservation
    plt.subplot(2, 1, 2)
    plt.plot(results["timesteps"], results["sedd_preservation"], 'b-', label='SEDD')
    plt.plot(results["timesteps"], results["uniform_preservation"], 'r-', label='Uniform')
    plt.xlabel('Timestep')
    plt.ylabel('Preserved Tokens (%)')
    plt.title('Token Preservation by Schedule Type')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Results saved to {output_file}")

def main():
    # Configuration
    vocab_size = 50257
    num_timesteps = 1000
    token_lengths = [100, 500, 1000]
    benchmark_timesteps = [1, 100, 200, 400, 600, 800, 999]
    
    # Initialize noisers
    sedd_noiser = TokenNoiser(vocab_size=vocab_size, num_timesteps=num_timesteps)
    uniform_noiser = None  # Will be imported from uniform_schedule.py
    
    try:
        from benchmarks.uniform_schedule import UniformNoiser
        uniform_noiser = UniformNoiser(vocab_size=vocab_size, num_timesteps=num_timesteps)
    except ImportError:
        print("Error: Unable to import UniformNoiser. Make sure it's implemented in benchmarks/uniform_schedule.py")
        return
    
    print("Starting benchmarks...")
    
    # Run benchmarks for different token lengths
    all_results = {}
    for length in token_lengths:
        print(f"Benchmarking with token length: {length}")
        tokens = generate_sample_tokens(length, vocab_size)
        
        results = benchmark_noising(
            tokens=tokens,
            timesteps=benchmark_timesteps,
            noiser_sedd=sedd_noiser,
            noiser_uniform=uniform_noiser
        )
        
        # Save results
        all_results[f"length_{length}"] = results
        
        # Plot individual results
        plot_results(results, f"benchmark_results_length_{length}.png")
    
    # Save all results to JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("Benchmarks completed! Results saved to benchmark_results.json and PNG files.")

if __name__ == "__main__":
    main()
