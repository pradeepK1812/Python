import numpy as np
import matplotlib.pyplot as plt
import math
import os

class BayesianKDECoin:
    """
    Bayesian coin posterior estimation using a nonparametric KDE-style approach.
    Posterior is computed on a grid: prior(p) * likelihood(data | p),
    then smoothed with a Gaussian kernel (no Beta assumption).
    """

    def __init__(self, prior_func=None, true_p=0.7, seed=0):
        """
        prior_func: callable prior_func(p) giving prior density for p ∈ [0,1]
                    If None => uniform prior
        true_p: probability used to simulate coin flips (hidden ground truth)
        seed: RNG seed for reproducibility
        """
        self.prior_func = prior_func if prior_func is not None else (lambda p: np.ones_like(p))
        self.true_p = true_p
        np.random.seed(seed)

    def simulate_flips(self, n):
        """Simulate 'H'/'T' coin flips from true_p."""
        return np.random.choice(["H", "T"], size=n, p=[self.true_p, 1 - self.true_p])

    def likelihood_on_grid(self, p_grid, flips):
        """Compute likelihood p^heads * (1-p)^tails on p_grid."""
        heads = np.sum(flips == "H")
        tails = np.sum(flips == "T")

        eps = 1e-12
        p_safe = np.clip(p_grid, eps, 1 - eps)
        return (p_safe ** heads) * ((1 - p_safe) ** tails)

    def gaussian_kernel(self, x, bandwidth):
        """Gaussian kernel values."""
        return np.exp(-0.5 * (x / bandwidth) ** 2) / (bandwidth * math.sqrt(2 * math.pi))

    def smooth_density(self, grid, density, bandwidth):
        """Gaussian convolution smoothing."""
        dx = grid[1] - grid[0]
        kernel_extent = int(min(len(grid)//2, max(1, int(np.ceil(4 * bandwidth / dx)))))
        kernel_x = np.linspace(-kernel_extent * dx, kernel_extent * dx, 2 * kernel_extent + 1)
        kernel = self.gaussian_kernel(kernel_x, bandwidth)
        kernel /= kernel.sum()

        smoothed = np.convolve(density, kernel, mode='same')
        smoothed = np.clip(smoothed, 0, None)

        area = np.trapz(smoothed, grid)
        if area > 0:
            smoothed /= area
        return smoothed

    def posterior_grid(self, flips, grid_points=1000):
        """Posterior grid before smoothing."""
        grid = np.linspace(0, 1, grid_points)
        prior_vals = self.prior_func(grid)
        like = self.likelihood_on_grid(grid, flips)
        unnorm = prior_vals * like

        area = np.trapz(unnorm, grid)
        if area > 0:
            unnorm /= area

        return grid, unnorm

    def kde_posterior(self, flips, grid_points=1000, bandwidth=0.02,
                      show_raw=False, save_path="bayes_kde.png"):
        """Compute KDE posterior & plot."""
        grid, post = self.posterior_grid(flips, grid_points)
        smoothed = self.smooth_density(grid, post, bandwidth)

        mean_est = np.trapz(grid * smoothed, grid)
        map_est = grid[np.argmax(smoothed)]

        fig, ax = plt.subplots(figsize=(10, 6))
        if show_raw:
            ax.plot(grid, post, "--", alpha=0.6, label="Raw posterior")

        ax.plot(grid, smoothed, label=f"KDE posterior (bw={bandwidth})")

        prior_vals = self.prior_func(grid)
        prior_area = np.trapz(prior_vals, grid)
        if prior_area > 0:
            ax.plot(grid, prior_vals / prior_area, ":", alpha=0.7, label="Prior")

        ax.axvline(mean_est, linestyle=":", label=f"Posterior mean={mean_est:.3f}")
        ax.axvline(map_est, linestyle="--", label=f"MAP={map_est:.3f}")

        ax.set_title("Bayesian KDE Posterior Estimate for P(heads)")
        ax.set_xlabel("p")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend()

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "grid": grid,
            "raw_posterior": post,
            "smoothed_posterior": smoothed,
            "mean": mean_est,
            "map": map_est,
            "save_path": os.path.abspath(save_path),
        }


if __name__ == "__main__":
    print("Running KDE Bayesian Coin demo...")
    model = BayesianKDECoin(true_p=0.7, seed=42)
    flips = model.simulate_flips(30)
    result = model.kde_posterior(flips, show_raw=True, save_path="bayes_kde_demo.png")

    heads = np.sum(flips == "H")
    tails = np.sum(flips == "T")
    print(f"Posterior mean = {result['mean']:.4f}, MAP = {result['map']:.4f}")
    print(f"Saved plot to: {result['save_path']}")
    print(f"Flips: {heads}H, {tails}T")
