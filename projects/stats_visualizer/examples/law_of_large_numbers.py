import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def demonstrate_lln(n_samples=5000, save_path="lln_demo.png"):
    """
    Demonstrates the Law of Large Numbers (LLN) using repeated sampling.
    
    LLN states that as sample size grows, the sample mean converges to the true mean.
    """

    # True distribution: Normal(μ=50, σ=10)
    true_mean = 50
    true_std = 10

    # Generate random samples
    samples = np.random.normal(loc=true_mean, scale=true_std, size=n_samples)

    # Running sample mean
    running_mean = np.cumsum(samples) / np.arange(1, n_samples + 1)

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(running_mean, label="Running Sample Mean", color="blue")
    plt.axhline(true_mean, color="red", linestyle="--", linewidth=2,
                label=f"True Mean = {true_mean}")

    plt.title("Law of Large Numbers Demonstration")
    plt.xlabel("Number of Samples (n)")
    plt.ylabel("Sample Mean")
    plt.grid(True, alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n LLN plot saved to: {os.path.abspath(save_path)}")
    print(f"Final running mean after {n_samples} samples: {running_mean[-1]:.4f}")
    print(f"True mean: {true_mean}")
    print("Difference:", abs(running_mean[-1] - true_mean))


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate the Law of Large Numbers (LLN)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="Number of samples to draw (default: 5000)"
    )

    args, _ = parser.parse_known_args()
    demonstrate_lln(args.n_samples)


if __name__ == "__main__":
    main()
