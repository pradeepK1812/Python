import numpy as np
import matplotlib.pyplot as plt
from projects.stats_visualizer.stats_visualizer.bayesian_coin import BayesianCoin
from projects.stats_visualizer.stats_visualizer.bayesian_kde_coin import BayesianKDECoin

def compare_beta_vs_kde(true_p=0.7, seed=42, n_flips=30):
    """Compare Beta posterior (conjugate) vs KDE posterior (nonparametric)."""
    
    np.random.seed(seed)
    flips = np.random.choice(["H", "T"], size=n_flips, p=[true_p, 1 - true_p])
    heads = np.sum(flips == "H")
    tails = np.sum(flips == "T")
    print(f"Simulated {n_flips} flips: {heads} Heads, {tails} Tails")

    # --- Beta Model Posterior ---
    beta_model = BayesianCoin(prior_a=1, prior_b=1, true_p=true_p)
    grid = np.linspace(0, 1, 1000)
    a_post, b_post = beta_model.update_posterior(flips)
    pdf_beta = np.array([beta_model.beta_pdf(xx, a_post, b_post) for xx in grid])
    pdf_beta /= np.trapz(pdf_beta, grid)

    # --- KDE Model Posterior ---
    kde_model = BayesianKDECoin(true_p=true_p, seed=seed)
    grid_kde, kde_post = kde_model.posterior_grid(flips)
    kde_smoothed = kde_model.smooth_density(grid_kde, kde_post, bandwidth=0.02)

    # --- Plot Comparison ---
    plt.figure(figsize=(10, 6))

    """plt.plot(grid, pdf_beta, color="#1f77b4", linewidth=2.5,
             label=f"Beta Posterior (α={a_post}, β={b_post})")

    plt.plot(grid_kde, kde_post, color="#d62728", linestyle="--", linewidth=1.8, alpha=0.6,
             label="KDE Raw Posterior")

    plt.plot(grid_kde, kde_smoothed, color="#2ca02c", linewidth=2,
             label="KDE Smoothed Posterior (bw=0.02)")

    plt.axvline(true_p, color="black", linestyle=":", linewidth=1.5, label=f"True p={true_p}")"""
    #plt.plot(grid, pdf_beta, color="#1f77b4", linewidth=2.5,
     #    label=f"Beta Posterior (α={a_post}, β={b_post})")
    #offset = 0.40
    #plt.plot(grid_kde, kde_post + offset, color="#d62728", linestyle="--", linewidth=1.8, alpha=0.9,
     #    label=f"KDE Raw Posterior (+{offset} offset)")
    #plt.fill_between(grid_kde, kde_post, color="#ff7f0e", alpha=0.25, label="KDE Raw Posterior (area)")
    #plt.plot(grid_kde, kde_smoothed, color="#2ca02c", linewidth=2.3,
     #    label="KDE Smoothed Posterior (bw=0.02)")
    vertical_offset = 0.3 * np.max(pdf_beta)
    plt.fill_between(grid_kde, kde_post + vertical_offset, vertical_offset,
                 color="orange", alpha=0.25, label="KDE Raw Posterior (vert. offset)")

    #plt.fill_between(grid_kde, 0, kde_post, color="orange", alpha=0.2, label="KDE Raw Posterior")
    plt.plot(grid, pdf_beta, color="blue", linewidth=2.2, label=f"Beta Posterior (α={a_post}, β={b_post})")
    plt.plot(grid_kde, kde_smoothed, color="red", linewidth=2, label="KDE Smoothed Posterior (bw=0.02)")

    plt.axvline(true_p, color="black", linestyle=":", linewidth=1.6, label=f"True p={true_p}")

    plt.xlabel("Probability of Heads (p)")
    plt.ylabel("Density")
    plt.title("Comparison: Beta vs KDE Posterior for Coin Flip")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = "compare_beta_kde_with_colors.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Comparison plot saved to {save_path}")

if __name__ == "__main__":
    compare_beta_vs_kde()
