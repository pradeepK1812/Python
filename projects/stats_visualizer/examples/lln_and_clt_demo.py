#!/usr/bin/env python3
"""
LLN + CLT demo.

Generates:
 - lln_demo.png   : running sample mean for one long sequence (Law of Large Numbers)
 - clt_demo.png   : histograms of sample means for multiple sample sizes with Normal overlays (Central Limit Theorem)

Usage:
    python -m projects.stats_visualizer.examples.lln_and_clt_demo --n_samples 8000 --num_trials 5000 --seed 42
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def demonstrate_lln(samples, dist_sampler, dist_name, true_mean, save_path="lln_demo.png"):
    """Plot running sample mean (LLN)."""
    data = dist_sampler(samples)
    running_mean = np.cumsum(data) / np.arange(1, samples + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(running_mean, label="Running sample mean", linewidth=1)
    plt.axhline(true_mean, color="red", linestyle="--", linewidth=2, label=f"True mean = {true_mean:.4f}")
    plt.title(f"Law of Large Numbers — running mean (parent: {dist_name})")
    plt.xlabel("Number of samples")
    plt.ylabel("Sample mean")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved LLN plot to: {os.path.abspath(save_path)}")


def demonstrate_clt(num_trials, sample_sizes, dist_sampler, dist_name, parent_mean, parent_var, save_path="clt_demo.png"):
    """
    For each n in sample_sizes:
      - draw num_trials samples of size n
      - compute sample means
      - plot histogram of sample means
      - overlay Normal( mean=parent_mean, sd = sqrt(parent_var / n) )
    """
    n_sizes = len(sample_sizes)
    cols = min(3, n_sizes)
    rows = (n_sizes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)  # flatten

    # For reproducibility use numpy Random Generator inside
    rng = np.random.default_rng()

    for idx, n in enumerate(sample_sizes):
        # Generate sample means: shape (num_trials,)
        # Draw num_trials * n random variates and reshape
        draws = dist_sampler(num_trials * n)
        draws = draws.reshape(num_trials, n)
        sample_means = draws.mean(axis=1)

        ax = axes[idx]
        # Histogram
        ax.hist(sample_means, bins=50, density=True, alpha=0.6, color="skyblue", label=f"Empirical means (n={n})")

        # Overlay normal pdf
        clt_mean = parent_mean
        clt_std = np.sqrt(parent_var / n)
        x_min, x_max = ax.get_xlim()
        x = np.linspace(x_min, x_max, 300)
        ax.plot(x, norm.pdf(x, loc=clt_mean, scale=clt_std), color="darkred", linewidth=2,
                label=f"Normal approx\nμ={clt_mean:.3f}, σ={clt_std:.3f}")

        ax.set_title(f"CLT: sample means (n={n})")
        ax.set_xlabel("sample mean")
        ax.set_ylabel("density")
        ax.legend()
        ax.grid(alpha=0.2)

    # hide unused subplots
    for j in range(idx + 1, rows * cols):
        axes[j].axis("off")

    plt.suptitle(f"Central Limit Theorem — parent distribution: {dist_name}\n(num_trials={num_trials})", y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved CLT plot to: {os.path.abspath(save_path)}")


def main():
    parser = argparse.ArgumentParser(description="LLN and CLT demonstration")
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Number of samples to draw for LLN running mean (default: 5000)")
    parser.add_argument("--num_trials", type=int, default=2000,
                        help="Number of repeated trials to build sampling distribution for CLT (default: 2000)")
    parser.add_argument("--sample_sizes", type=int, nargs="+", default=[1, 4, 16, 64],
                        help="List of sample sizes n to test for CLT (default: 1 4 16 64)")
    parser.add_argument("--dist", choices=["exponential", "uniform", "bernoulli", "poisson"], default="exponential",
                        help="Parent distribution to sample from (default: exponential)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: None)")
    parser.add_argument("--out_dir", type=str, default=".", help="Directory to save images")
    # FIX: Use parse_known_args() to safely ignore arguments like -f 
    # that are automatically passed by the Colab/Jupyter kernel.
    args, _ = parser.parse_known_args()

    rng = np.random.default_rng(args.seed)

    # define sampler, mean and variance for each supported distribution
    if args.dist == "exponential":
        lam = 1.0  # rate
        sampler = lambda m: rng.exponential(scale=1.0 / lam, size=m)
        parent_mean = 1.0 / lam
        parent_var = 1.0 / (lam ** 2)
        dist_name = f"Exponential(λ={lam})"
    elif args.dist == "uniform":
        a, b = 0.0, 2.0
        sampler = lambda m: rng.uniform(low=a, high=b, size=m)
        parent_mean = 0.5 * (a + b)
        parent_var = (b - a) ** 2 / 12.0
        dist_name = f"Uniform({a},{b})"
    elif args.dist == "bernoulli":
        p = 0.3
        sampler = lambda m: rng.binomial(1, p, size=m)
        parent_mean = p
        parent_var = p * (1 - p)
        dist_name = f"Bernoulli(p={p})"
    elif args.dist == "poisson":
        lam = 2.0
        sampler = lambda m: rng.poisson(lam, size=m)
        parent_mean = lam
        parent_var = lam
        dist_name = f"Poisson(λ={lam})"
    else:
        raise SystemExit("Unknown distribution")

    # make output directory
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # LLN demo
    lln_path = os.path.join(out_dir, f"lln_{args.dist}.png")
    demonstrate_lln(args.n_samples, lambda m: sampler(m), dist_name, parent_mean, save_path=lln_path)

    # CLT demo
    clt_path = os.path.join(out_dir, f"clt_{args.dist}.png")
    demonstrate_clt(args.num_trials, args.sample_sizes, lambda m: sampler(m), dist_name, parent_mean, parent_var, save_path=clt_path)


if __name__ == "__main__":
    main()
