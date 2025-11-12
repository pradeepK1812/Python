"""
Example script: Demonstrates Bayesian KDE coin posterior estimation.

Usage:
    python -m projects.stats_visualizer.examples.run_kde_demo
"""

import numpy as np
from projects.stats_visualizer.stats_visualizer.bayesian_kde_coin import BayesianKDECoin


def main():
    # Initialize model with true coin probability = 0.7
    model = BayesianKDECoin(true_p=0.7, seed=42)

    # Simulate 30 coin flips
    flips = model.simulate_flips(30)
    print(f"Simulated flips: {''.join(flips)}")

    # Compute and plot the KDE posterior
    result = model.kde_posterior(
        flips,
        grid_points=1000,
        bandwidth=0.02,
        show_raw=True,
        save_path="bayes_kde_demo.png",
    )

    print("\n===== Results =====")
    print(f"Posterior mean  : {result['mean']:.4f}")
    print(f"MAP estimate    : {result['map']:.4f}")
    print(f"Saved plot path : {result['save_path']}")

    heads = np.sum(flips == "H")
    tails = np.sum(flips == "T")
    print(f"Heads={heads}, Tails={tails}")


if __name__ == "__main__":
    main()
