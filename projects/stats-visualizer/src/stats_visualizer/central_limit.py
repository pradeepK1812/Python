import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def clt_visualization(dist_data, dist_name, sample_size, num_samples):
    sample_means = [np.mean(np.random.choice(dist_data, sample_size)) for _ in range(num_samples)]

    plt.hist(sample_means, bins=30, density=True, alpha=0.6, label="Sample Means")

    # Overlay True Normal Curve ✅
    mu = np.mean(dist_data)
    sigma = np.std(dist_data, ddof=1) / np.sqrt(sample_size)
    x = np.linspace(min(sample_means), max(sample_means), 200)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r-', label="True Normal Curve")

    plt.title(f"CLT with {dist_name} Distribution\nSample Size n={sample_size}, Samples={num_samples}")
    plt.legend()
    plt.grid(True)


def get_distribution(dist_type, size=100000):
    if dist_type == "uniform":
        return np.random.uniform(0, 1, size)
    elif dist_type == "exponential":
        return np.random.exponential(1, size)
    elif dist_type == "bernoulli":
        return np.random.binomial(1, 0.5, size)  # p = 0.5
    elif dist_type == "normal":
        return np.random.normal(0, 1, size)
    else:
        raise ValueError("Unsupported distribution type")


if __name__ == "__main__":
    for dist in ["uniform", "exponential", "bernoulli"]:
        data = get_distribution(dist)
        clt_visualization(data, dist_name=dist.capitalize(),
                          sample_size=30, num_samples=5000)
        plt.savefig(f"clt_{dist}.png")
        plt.show()
