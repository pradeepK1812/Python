import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t, norm

def plot_t_convergence(df_values=[1, 3, 10, 30], output_path="t_convergence.png"):
    x = np.linspace(-4, 4, 400)
    normal_pdf = norm.pdf(x)

    plt.figure(figsize=(10, 6))

    # Plot Normal distribution
    plt.plot(x, normal_pdf, label="Normal Distribution (µ=0, σ=1)", linewidth=2)

    # Plot T distributions for each df
    for df in df_values:
        t_pdf = t.pdf(x, df)
        plt.plot(x, t_pdf, '--', linewidth=1.8, label=f"t-dist (df={df})")

    plt.title("Convergence of t-distribution to Normal distribution")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # ✅ Save inside function
    plt.savefig(output_path)
    return output_path


if __name__ == "__main__":
    output = plot_t_convergence()
    print(f"Plot saved → {output}")
    plt.show()
