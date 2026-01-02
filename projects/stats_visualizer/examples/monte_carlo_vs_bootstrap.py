import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
np.random.seed(42)

TRUE_MEAN = 50
TRUE_STD = 10
SAMPLE_SIZE = 30
N_SIMULATIONS = 10_000

# -----------------------------
# Step 1: Observed sample
# -----------------------------
observed_data = np.random.normal(TRUE_MEAN, TRUE_STD, SAMPLE_SIZE)

# -----------------------------
# Step 2: Monte Carlo Simulation
# -----------------------------
mc_means = np.array([
    np.random.normal(TRUE_MEAN, TRUE_STD, SAMPLE_SIZE).mean()
    for _ in range(N_SIMULATIONS)
])

# -----------------------------
# Step 3: Bootstrapping
# -----------------------------
#bootstrap_means = np.array([
 #   np.random.choice(observed_data, size=SAMPLE_SIZE, replace=True).mean()
  #  for _ in range(N_SIMULATIONS)
#])
bootstrap_means = []

for _ in range(N_SIMULATIONS):
    # THIS is the bootstrap resampling step
    resample = np.random.choice(
        observed_data,
        size=SAMPLE_SIZE,
        replace=True
    )

    # statistic on resample
    bootstrap_means.append(resample.mean())

bootstrap_means = np.array(bootstrap_means)

# -----------------------------
# Step 4: Plot (save to PNG)
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"monte_carlo_vs_bootstrap_{timestamp}.png"

print(f"Saving plot as: {filename}")

plt.figure(figsize=(10, 6))

plt.hist(mc_means, bins=40, density=True, alpha=0.6, label="Monte Carlo Means")
plt.hist(bootstrap_means, bins=40, density=True, alpha=0.6, label="Bootstrap Means")

plt.axvline(TRUE_MEAN, linestyle="--", linewidth=2, label="True Mean")
plt.axvline(observed_data.mean(), linestyle=":", linewidth=2, label="Observed Mean")

plt.title("Monte Carlo vs Bootstrapping (Sampling Distribution of Mean)")
plt.xlabel("Sample Mean")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.savefig(filename, dpi=150)
plt.close()

# -----------------------------
# Step 5: Numerical comparison
# -----------------------------
print("\n===== Monte Carlo =====")
print(f"Mean Estimate  : {mc_means.mean():.3f}")
print(f"Std Error     : {mc_means.std():.3f}")

print("\n===== Bootstrap =====")
print(f"Mean Estimate  : {bootstrap_means.mean():.3f}")
print(f"Std Error     : {bootstrap_means.std():.3f}")

# -----------------------------
# Step 6: 95% Confidence Intervals
# -----------------------------
mc_ci = np.percentile(mc_means, [2.5, 97.5])
bs_ci = np.percentile(bootstrap_means, [2.5, 97.5])

print("\n===== 95% Confidence Intervals =====")
print(f"Monte Carlo CI : {mc_ci}")
print(f"Bootstrap CI   : {bs_ci}")
