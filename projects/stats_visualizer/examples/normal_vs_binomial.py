import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

def calculate_probabilities(n_flips, p_head=0.5):
    """
    Calculates the exact binomial probability and the normal approximation
    for getting between 190 and 230 heads.

    Args:
        n_flips (int): The total number of coin flips (trials).
        p_head (float): The probability of getting a head on a single flip.

    Returns:
        tuple: (binomial_prob, normal_prob, mean, std_dev)
    """
    # --- Binomial Distribution Parameters ---
    n = n_flips
    p = p_head

    # 1. Exact Binomial Probability P(190 <= X <= 230)
    # P(a <= X <= b) = CDF(b) - CDF(a-1)
    
    lower_bound = 190
    upper_bound = 230

    # Ensure bounds are valid
    if lower_bound > n or upper_bound < 0:
        return 0, 0, 0, 0

    # CDF(230) - CDF(189)
    binomial_prob = binom.cdf(upper_bound, n, p) - binom.cdf(lower_bound - 1, n, p)

    # --- Normal Approximation Parameters ---
    # Mean (mu) = n * p
    mean = n * p
    # Standard Deviation (sigma) = sqrt(n * p * (1 - p))
    std_dev = np.sqrt(n * p * (1 - p))

    # 2. Normal Approximation Probability using Continuity Correction
    # P(190 <= X <= 230) approx P(189.5 <= Y <= 230.5)
    
    z_lower = (lower_bound - 0.5 - mean) / std_dev
    z_upper = (upper_bound + 0.5 - mean) / std_dev
    
    # Calculate probability using the Normal CDF (Standard Normal - Z-score)
    normal_prob = norm.cdf(z_upper) - norm.cdf(z_lower)

    return binomial_prob, normal_prob, mean, std_dev

def plot_distributions(n_flips, mean, std_dev, p_head=0.5):
    """
    Generates a plot comparing the Binomial PMF and the Normal PDF.
    """
    n = n_flips
    p = p_head
    
    # The range of possible number of heads (0 to n)
    k = np.arange(0, n + 1)
    
    # 1. Binomial Distribution (PMF - Probability Mass Function)
    binomial_pmf = binom.pmf(k, n, p)
    
    # 2. Normal Distribution (PDF - Probability Density Function)
    # Using the same range k for the x-axis, calculate the PDF
    normal_pdf = norm.pdf(k, mean, std_dev)
    save_path = "normal_vs_binomial.png"
    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    
    # Plot Binomial Distribution (Discrete bars)
    plt.bar(k, binomial_pmf, color='skyblue', alpha=0.6, label='Binomial PMF')
    
    # Plot Normal Distribution (Continuous line)
    plt.plot(k, normal_pdf, color='darkred', linewidth=2, label='Normal PDF Approximation')
    
    # Highlight the target range (190 to 230 heads)
    highlight_range = np.arange(190, 231)
    highlight_pmf = binom.pmf(highlight_range, n, p)
    plt.bar(highlight_range, highlight_pmf, color='gold', alpha=1.0, label='Target Range (190-230)')

    plt.title(f'Binomial Distribution (n={n}, p={p}) vs. Normal Approximation')
    plt.xlabel('Number of Heads (k)')
    plt.ylabel('Probability / Density')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n📊 Plot saved to: {os.path.abspath(save_path)}")

def main():
    """
    Main function to parse command-line arguments and run the comparison.
    Handles extra arguments passed by Jupyter/Colab kernels.
    """
    parser = argparse.ArgumentParser(
        description="Compare Binomial and Normal Distributions for coin flips.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--n_flips', 
        type=int, 
        default=400, 
        help="Number of coin flips (trials). Default is 400."
    )
    
    # FIX: Use parse_known_args() to safely ignore arguments like -f 
    # that are automatically passed by the Colab/Jupyter kernel.
    args, unknown = parser.parse_known_args()
    n_flips = args.n_flips

    if n_flips <= 0:
        print("Error: The number of flips must be a positive integer.")
        return

    # Check for Normal Approximation requirement (n*p >= 5 and n*(1-p) >= 5)
    if n_flips * 0.5 < 5:
        print(f"Warning: n={n_flips} is too small for a good Normal approximation (requires np >= 5).")

    binomial_prob, normal_prob, mean, std_dev = calculate_probabilities(n_flips)

    print(f"\n--- Distribution Comparison (N={n_flips}, P=0.5) ---")
    print(f"Approximation Check (Mean/SD):")
    print(f"  Mean (μ) = {mean:.2f}")
    print(f"  Standard Deviation (σ) = {std_dev:.2f}")
    
    print("\nProbability of getting 190 to 230 Heads:")
    print(f"  Exact Binomial Probability: {binomial_prob * 100:.4f}%")
    print(f"  Normal Approximation Probability: {normal_prob * 100:.4f}% (Using Continuity Correction)")

    plot_distributions(n_flips, mean, std_dev)

if __name__ == '__main__':
    main()
