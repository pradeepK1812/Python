import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, expon

def compare_gamma_exp_normal(shape=0.4, rate=1.2, save_path="gamma_exp_normal.png"):
    """
    Compare Gamma vs Exponential vs Normal distribution for rare-event waiting times.
    
    Gamma(shape < 1) -> heavy right tail
    Exponential(rate) -> memoryless waiting time
    Normal -> poor approximation but included for contrast
    """

    # Convert rate λ to scale θ for SciPy
    scale = 1.0 / rate

    # Theoretical mean and variance of Gamma
    mean = shape * scale
    var = shape * (scale ** 2)
    std_dev = np.sqrt(var)

    print(f"Gamma(shape={shape}, rate={rate}) → mean={mean:.4f}, std={std_dev:.4f}")

    # Grid for plotting
    x = np.linspace(0.0001, mean + 8 * std_dev, 800)

    # 1. Gamma PDF
    gamma_pdf = gamma.pdf(x, a=shape, scale=scale)

    # 2. Exponential PDF (rate λ)
    exp_pdf = expon.pdf(x, scale=scale)

    # 3. Normal approximation PDF
    normal_pdf = norm.pdf(x, loc=mean, scale=std_dev)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, gamma_pdf, label=f"Gamma (shape={shape})", linewidth=2)
    plt.plot(x, exp_pdf, label=f"Exponential (rate={rate})", linestyle="-.", linewidth=2)
    plt.plot(x, normal_pdf, label="Normal Approximation", linestyle="--", linewidth=2)

    plt.title("Gamma vs Exponential vs Normal Distribution\n(Rare Event Waiting Time Example)")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    compare_gamma_exp_normal()
