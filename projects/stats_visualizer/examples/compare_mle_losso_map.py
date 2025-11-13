import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def generate_data(n=50, a_true=2.0, b_true=0.5, noise_std=0.3, seed=42):
    np.random.seed(seed)
    x = np.linspace(0, 5, n)
    y = a_true * x + b_true + np.random.normal(0, noise_std, size=n)
    return x, y

def mle_estimate(x, y):
    """Ordinary least squares (MLE)."""
    A = np.vstack([x, np.ones_like(x)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a_mle, b_mle = coeffs
    return a_mle, b_mle

def lasso_estimate(x, y, alpha=0.05):
    """Lasso regularized MLE."""
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    model.fit(x.reshape(-1, 1), y)
    return model.coef_[0], model.intercept_

def map_estimate(x, y, tau=1.0, b_prior_mean=0.1):
    """
    MAP estimation with Gaussian priors:
    a ~ N(0, tau^2), b ~ N(b_prior_mean, tau^2)
    """
    A = np.vstack([x, np.ones_like(x)]).T
    # Posterior precision = A^T A + (1/tau^2) I
    precision = A.T @ A + (1 / tau**2) * np.eye(2)
    rhs = A.T @ y + (1 / tau**2) * np.array([0, b_prior_mean])
    params = np.linalg.solve(precision, rhs)
    a_map, b_map = params
    return a_map, b_map

def plot_comparison(x, y, a_mle, b_mle, a_lasso, b_lasso, a_map, b_map):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=40, color='black', alpha=0.7, label="Data")
    x_line = np.linspace(x.min(), x.max(), 100)

    plt.plot(x_line, a_mle*x_line + b_mle, 'b-', label=f"MLE: y={a_mle:.2f}x+{b_mle:.2f}")
    plt.plot(x_line, a_lasso*x_line + b_lasso, 'r--', label=f"Lasso: y={a_lasso:.2f}x+{b_lasso:.2f}")
    plt.plot(x_line, a_map*x_line + b_map, 'g-.', label=f"MAP: y={a_map:.2f}x+{b_map:.2f}")

    plt.title("Comparison of MLE, Lasso, and MAP Estimates")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("compare_mle_lasso_map.png", dpi=300)
    print("Plot saved in compare_mle_lasso_map.png")
    plt.show()

if __name__ == "__main__":
    x, y = generate_data()
    a_mle, b_mle = mle_estimate(x, y)
    a_lasso, b_lasso = lasso_estimate(x, y, alpha=0.05)
    a_map, b_map = map_estimate(x, y, tau=1.0, b_prior_mean=0.1)

    print(f"MLE:   a={a_mle:.3f}, b={b_mle:.3f}")
    print(f"Lasso: a={a_lasso:.3f}, b={b_lasso:.3f}")
    print(f"MAP:   a={a_map:.3f}, b={b_map:.3f}")

    plot_comparison(x, y, a_mle, b_mle, a_lasso, b_lasso, a_map, b_map)
