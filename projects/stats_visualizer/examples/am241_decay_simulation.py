import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, expon

def simulate_am241_clicks(num_clicks=10220, rate_per_sec=1.7, interval_sec=10, seed=42):
    """
    Simulate Am-241 alpha particle decay clicks.
    - num_clicks: total number of Geiger counter clicks (from Rice dataset example)
    - rate_per_sec: expected mean click rate λ (clicks per second)
    - interval_sec: bin width for counts
    - seed: RNG seed for reproducibility
    """
    np.random.seed(seed)

    # Simulate inter-arrival times using exponential distribution
    inter_arrival_times = np.random.exponential(1 / rate_per_sec, size=num_clicks)
    timestamps = np.cumsum(inter_arrival_times)

    # Bin data in 10-second intervals
    max_time = timestamps[-1]
    bins = np.arange(0, max_time + interval_sec, interval_sec)
    counts, _ = np.histogram(timestamps, bins=bins)

    return inter_arrival_times, counts, rate_per_sec, interval_sec


def plot_am241_distributions(inter_arrival_times, counts, rate, interval):
    """Plot Poisson (counts per interval) and Exponential (interarrival times)."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # --- (1) Poisson Distribution: counts per 10s interval ---
    mean_counts = rate * interval
    x_vals = np.arange(0, np.max(counts) + 1)
    poisson_pmf = poisson.pmf(x_vals, mean_counts)

    axs[0].hist(counts, bins=np.arange(np.min(counts), np.max(counts) + 1) - 0.5,
                density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Observed')
    axs[0].plot(x_vals, poisson_pmf, 'r--', lw=2, label=f'Poisson(λ={mean_counts:.1f})')
    axs[0].set_xlabel("Clicks per 10s interval")
    axs[0].set_ylabel("Probability")
    axs[0].set_title("Am-241 Decay Counts (Poisson)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # --- (2) Exponential Distribution: time between clicks ---
    lam = rate
    x = np.linspace(0, np.max(inter_arrival_times), 200)
    exp_pdf = expon.pdf(x, scale=1 / lam)

    axs[1].hist(inter_arrival_times, bins=50, density=True,
                alpha=0.6, color='lightgreen', edgecolor='black', label='Observed')
    axs[1].plot(x, exp_pdf, 'r--', lw=2, label=f'Exp(λ={lam:.2f})')
    axs[1].set_xlabel("Time between clicks (s)")
    axs[1].set_ylabel("Density")
    axs[1].set_title("Interarrival Times (Exponential)")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.suptitle("Americium-241 Decay Simulation (Radioactive Clicks)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    print("Plot saved to am241_decay_simulation.png")
    plt.savefig("am241_decay_simulation.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    inter_arrival_times, counts, rate, interval = simulate_am241_clicks()
    plot_am241_distributions(inter_arrival_times, counts, rate, interval)
