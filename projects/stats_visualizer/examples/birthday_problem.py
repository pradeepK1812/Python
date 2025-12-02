import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def analytic_birthday_probability(n):
    """
    Computes the analytic probability that at least two people
    share a birthday in a group of size n.
    """
    if n > 365:
        return 1.0  # pigeonhole principle

    prob_unique = 1.0
    days = 365

    for i in range(n):
        prob_unique *= (days - i) / days # calculating the probabilities for no unique birthday for 1 to n people and final  probability will be product of all as all the cases are independent.

    return 1 - prob_unique


def simulate_birthday_probability(n, trials=5000, seed=42):
    """
    Monte Carlo simulation of the birthday problem.
    Randomly assigns each person a birthday, checks for duplicates.
    """
    np.random.seed(seed)
    count_shared = 0

    for _ in range(trials):
        birthdays = np.random.randint(0, 365, size=n)
        if len(np.unique(birthdays)) < n:
            count_shared += 1

    return count_shared / trials


def plot_birthday_curves(max_people=60, trials=5000, save_path="birthday_problem.png"):
    """
    Plots analytic vs simulated birthday problem probability curves.
    Saves the plot to a PNG file (for Codespaces).
    """
    people = np.arange(1, max_people + 1)
    analytic_probs = [analytic_birthday_probability(n) for n in people]
    sim_probs = [simulate_birthday_probability(n, trials=trials) for n in people]

    plt.figure(figsize=(10, 6))
    plt.plot(people, analytic_probs, label="Analytic Probability", linewidth=2)
    plt.plot(people, sim_probs, "o", markersize=3, label="Simulation", alpha=0.7)

    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.text(1, 0.52, "50% threshold", color="gray")

    plt.title("Birthday Paradox: Probability of Shared Birthday")
    plt.xlabel("Number of People (n)")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    print("Running Birthday Problem Demo...\n")
    #taking n as command line argument
    parser = argparse.ArgumentParser(description="Birthday Problem Demo")
    parser.add_argument(
        "-n", "--people",
        type=int,
        default=23,
        help="Number of people in the group (default = 23)"
    )

    args = parser.parse_args()
    n = args.people
    #n = 23
    analytic = analytic_birthday_probability(n)
    sim = simulate_birthday_probability(n)

    print(f"For n={n}:")
    print(f"Analytic probability = {analytic:.4f}")
    print(f"Simulated probability = {sim:.4f}")

    plot_birthday_curves()
