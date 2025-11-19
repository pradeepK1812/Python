"""
P-Hacking Demonstration
-----------------------

This script simulates repeated hypothesis testing as sample size increases.
A fair coin is flipped, and after each additional observation (from n=20 to 50)
we compute the p-value for the hypothesis that the coin is fair.

This demonstrates how "peeking" at the p-value multiple times
can lead to false positives by chance alone.
"""

import numpy as np
from scipy.stats import binomtest
import matplotlib.pyplot as plt
import argparse

def p_hacking_demo(min_n=20, max_n=50, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Simulate fair coin flips: 1 = heads, 0 = tails
    flips = np.random.choice([0, 1], size=max_n)

    ns = []
    p_values = []

    for n in range(min_n, max_n + 1):
        sample = flips[:n]
        heads = np.sum(sample)

        p = binomtest(
            heads,
            n,
            0.5,
            alternative="two-sided"
        ).pvalue

        ns.append(n)
        p_values.append(p)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(ns, p_values, marker="o")
    plt.axhline(0.05, linestyle="--")
    plt.xlabel("Number of coin flips (n)")
    plt.ylabel("P-value")
    plt.title("P-hacking demonstration: P-value vs sample size (fair coin)")
    plt.grid(True)
    plt.tight_layout()
    print("plot saved in my_p_hacking_demo.png")
    plt.savefig("my_p_hacking_demo.png", dpi=300, bbox_inches="tight")
    plt.close()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a p-hacking demonstration with optional parameters."
    )

    parser.add_argument(
        "--min_n",
        type=int,
        default=20,
        help="Minimum sample size to start calculating p-values."
    )

    parser.add_argument(
        "--max_n",
        type=int,
        default=50,
        help="Maximum sample size to stop calculating p-values."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    p_hacking_demo(
        min_n=args.min_n,
        max_n=args.max_n,
        seed=args.seed
    )

