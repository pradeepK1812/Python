import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm


def plot_t_to_normal_convergence(df_values=None, save_path=None):
    """
    Plot how the t-distribution approaches the standard normal distribution.
    """

    if df_values is None:
        df_values = [1, 5, 10, 30]

    x = np.linspace(-4, 4, 400)

    plt.figure(figsize=(10, 6))
    plt.plot(x, norm.pdf(x), label="Normal (μ=0, σ=1)", linewidth=2)

    for df in df_values:
        plt.plot(x, t.pdf(x, df), linestyle="--", label=f"t-dist (df={df})")

    plt.title("Convergence of t-Distribution to Normal")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        return save_path

    plt.show()
