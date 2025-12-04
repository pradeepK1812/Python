import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from math import comb, sqrt, pi, exp


def binomial_pmf(n, p):
    """Return array of Binomial PMF for k = 0..n."""
    k = np.arange(0, n + 1)
    pmf = np.array([comb(n, ki) * (p ** ki) * ((1 - p) ** (n - ki)) for ki in k])
    return k, pmf


def normal_pdf(x, mean, std):
    """Normal probability density function."""
    return (1 / (std * sqrt(2 * pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def probability_binomial_range(n, p, low, high):
    """Exact Binomial probability P(low ≤ X ≤ high)."""
    k = np.arange(low, high + 1)
    prob = np.sum([comb(n, ki) * (p ** ki) * ((1 - p) ** (n - ki)) for ki in k])
    return prob


def probability_normal_range(mean, std, low, high):
    """Normal approximation using numerical integration."""
    xs = np.linspace(low, high, 5000)
    pdf_vals = normal_pdf(xs, mean, std)
    return np.trapezoid(pdf_vals, xs)


def plot_binomial_vs_normal(n, p=0.5, low=190, high=230, save_path="binomial_vs_normal.png"):
    """Plot Binomial vs Normal approximation."""
    # Binomial
    k, binom_pmf = binomial_pmf(n, p)

    # Normal approximation
    mean = n * p
    std = sqrt(n * p * (1 - p))
    x = np.linspace(mean - 4 * std, mean + 4 * std, 500)
    norm_pdf = normal_pdf(x, mean, std)

    # Probabilities
    prob_binom = probability_binomial_range(n, p, low, high)
    prob_norm = probability_normal_range(mean, std, low, high)

    print(f"\n--- Probability Results for n={n} ---")
    print(f"Binomial P({low} ≤ X ≤ {high}) = {prob_binom:.4f}")
    print(f"Normal Approx P({low} ≤ X ≤ {high}) = {prob_norm:.4f}")

    # Plot
    plt.figure(figsize=(12, 6))

    # Binomial PMF bars
    plt.bar(k, binom_pmf, color="blue", alpha=0.5, label="Binomial PMF")

    # Normal PDF line (scaled so peak heights are visually comparable)
    plt.plot(x, norm_pdf * (max(binom_pmf) / max(norm_pdf)),
             color="red", linewidth=2, label="Normal Approximation")

    # Highlight region [low, high]
    region_k = np.arange(low, high + 1)
    region_pmf = binom_pmf[low:high + 1]
    plt.bar(region_k, region_pmf, color="green", alpha=0.6, label="Range Probability")

    plt.title(f"Binomial(n={n}, p={p}) vs Normal Approximation")
    plt.xlabel("Number of Heads")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n📊 Plot saved to: {os.path.abspath(save_path)}")


def main():
    parser = argparse.ArgumentParser(description="Binomial vs Normal distribution demo")
    parser.add_argument("-n", "--flips", type=int, default=400,
                        help="Number of coin flips (default = 400)")
    parser.add_argument("--low", type=int, default=190, help="Lower bound")
    parser.add_argument("--high", type=int, default=230, help="Upper bound")

    args = parser.parse_args()
    n = args.flips

    plot_binomial_vs_normal(n=n, low=args.low, high=args.high)


if __name__ == "__main__":
    main()
