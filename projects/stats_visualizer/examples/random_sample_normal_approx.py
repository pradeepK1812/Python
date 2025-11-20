import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
import argparse


def repeated_sampling_demo(
    N=15000, lam=10, sample_size=200, m=100, seed=None,
    filename="sample_means_vs_population.png"
):
    if seed is not None:
        np.random.seed(seed)

    # --- Step 1: Generate full population ---
    population = np.random.poisson(lam=lam, size=N)
    pop_mean = population.mean()
    pop_std = population.std()
    print(f"Population mean = {pop_mean:.3f}, std = {pop_std:.3f}")

    # --- Step 2: Draw m samples and compute means ---
    sample_means = np.array([
        np.random.choice(population, size=sample_size, replace=False).mean()
        for _ in range(m)
    ])

    print(f"Mean of sample means = {sample_means.mean():.3f}")
    print(f"Std of sample means  = {sample_means.std():.3f}")

    # --- Step 3: Compute theoretical normal approximation ---
    normal_mean = lam
    normal_sd = np.sqrt(lam / sample_size)

    x_min = min(sample_means.min(), population.min())
    x_max = max(sample_means.max(), population.max())
    x_vals = np.linspace(x_min, x_max, 400)
    normal_curve = norm.pdf(x_vals, loc=normal_mean, scale=normal_sd)

    # --- Step 4: Plot ---
    plt.figure(figsize=(10, 6))

    # Histogram: population
    plt.hist(population, bins=30, alpha=0.4, density=True,
             label=f"Population (N={N})")

    # Histogram: sample means
    plt.hist(sample_means, bins=20, alpha=0.8, density=True,
             label=f"{m} Sample Means (n={sample_size})")

    # Normal approximation curve
    plt.plot(
        x_vals,
        normal_curve,
        "r--",
        linewidth=2,
        label="Normal Approximation of Sample Means"
    )

    plt.title("Population vs Sample Means (with Normal Approximation)")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    print(f"Saving plot to: {filename}")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repeated sampling comparison of population vs sample means."
    )

    parser.add_argument("--N", type=int, default=15000)
    parser.add_argument("--lam", type=float, default=10)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--filename", type=str,
                        default="sample_means_vs_population.png")

    args = parser.parse_args()

    repeated_sampling_demo(
        N=args.N,
        lam=args.lam,
        sample_size=args.sample_size,
        m=args.m,
        seed=args.seed,
        filename=args.filename
    )
