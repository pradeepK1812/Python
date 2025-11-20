import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import argparse


def compare_population_sample(
    N=15000, lam=10, sample_size=200, seed=None, filename="population_vs_sample.png"
):
    if seed is not None:
        np.random.seed(seed)

    # --- Generate population ---
    population = np.random.poisson(lam=lam, size=N)

    # --- Draw sample ---
    sample = np.random.choice(population, size=sample_size, replace=False)

    # --- Compute stats ---
    pop_mean, pop_std = population.mean(), population.std()
    sample_mean, sample_std = sample.mean(), sample.std()

    print(f"Population mean = {pop_mean:.3f}, std = {pop_std:.3f}")
    print(f"Sample mean     = {sample_mean:.3f}, std = {sample_std:.3f}")

    # --- Plot distributions ---
    plt.figure(figsize=(10, 6))
    plt.hist(population, bins=30, alpha=0.5, density=True,
             label=f"Population (N={N})")
    plt.hist(sample, bins=30, alpha=0.7, density=True,
             label=f"Sample (n={sample_size})")

    # Theoretical Poisson curve
    x_vals = np.arange(0, max(population.max(), sample.max()) + 1)
    plt.plot(
        x_vals,
        poisson.pmf(x_vals, mu=lam),
        "k--",
        label="Poisson(λ) theoretical curve"
    )

    plt.title("Population vs Sample Distribution Comparison")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --- Save plot ---
    print(f"Saving plot to: {filename}")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Poisson population vs sample distribution."
    )

    parser.add_argument("--N", type=int, default=15000,
                        help="Population size (default: 15000)")
    parser.add_argument("--lam", type=float, default=10,
                        help="Lambda parameter for Poisson distribution (default: 10)")
    parser.add_argument("--sample_size", type=int, default=200,
                        help="Sample size drawn from population (default: 200)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None)")
    parser.add_argument("--filename", type=str, default="population_vs_sample.png",
                        help="Output PNG file name (default: population_vs_sample.png)")

    args = parser.parse_args()

    compare_population_sample(
        N=args.N,
        lam=args.lam,
        sample_size=args.sample_size,
        seed=args.seed,
        filename=args.filename
    )
