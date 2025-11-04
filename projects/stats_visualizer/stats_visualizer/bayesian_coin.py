import numpy as np
import matplotlib.pyplot as plt
import math
import os

class BayesianCoin:
    """
    A simple Bayesian model for estimating a coin's probability of landing heads
    using a Beta prior and coin flip observations.
    """

    def __init__(self, prior_a=1, prior_b=1, true_p=0.7, seed=0):
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.true_p = true_p
        np.random.seed(seed)

    def simulate_flips(self, n):
        """
        Simulate coin flips based on true probability.
        Returns a numpy array of 'H' and 'T'.
        """
        return np.random.choice(["H", "T"], size=n, p=[self.true_p, 1 - self.true_p])

    @staticmethod
    def beta_pdf(x, a, b):
        """
        Compute Beta PDF manually without SciPy.
        """
        B = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
        return (x ** (a - 1)) * ((1 - x) ** (b - 1)) / B

    def update_posterior(self, flips):
        """
        Update Beta distribution parameters after observing flips.
        """
        a = self.prior_a + np.sum(flips == "H")
        b = self.prior_b + np.sum(flips == "T")
        return a, b

    def plot_update(self, n_flips_list=[10, 30], save_path="bayesian_coin_update.png"):
        """
        Plots prior + posterior after given numbers of observations.
        Saves image to disk.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.linspace(0.001, 0.999, 500)

        # Prior distribution
        pdf_prior = [self.beta_pdf(xx, self.prior_a, self.prior_b) for xx in x]
        mean_prior = self.prior_a / (self.prior_a + self.prior_b)
        ax.plot(x, pdf_prior,
                label=f"Prior Beta({self.prior_a},{self.prior_b}) mean={mean_prior:.3f}")

        # Simulate flips once for consistent updating
        flips = self.simulate_flips(max(n_flips_list))

        for n in n_flips_list:
            a_post, b_post = self.update_posterior(flips[:n])
            pdf_post = [self.beta_pdf(xx, a_post, b_post) for xx in x]
            mean_post = a_post / (a_post + b_post)
            ax.plot(x, pdf_post,
                    label=f"{n} flips → Beta({a_post},{b_post}) mean={mean_post:.3f}")

        ax.set_title("Bayesian Update for Coin Probability (Beta Prior)")
        ax.set_xlabel("Probability of Heads")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True)

        # ✅ Save image instead of just showing
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ Plot saved to: {os.path.abspath(save_path)}")

        plt.close()

        # Print summary
        print("\nPosterior summaries:")
        for n in n_flips_list:
            a_post, b_post = self.update_posterior(flips[:n])
            heads = np.sum(flips[:n] == "H")
            print(f"{n} flips: {heads} Heads, {n - heads} Tails → Beta({a_post},{b_post}) mean={a_post / (a_post + b_post):.3f}")


# ✅ Example usage if file is executed directly
if __name__ == "__main__":
    print("Running Bayesian Coin Demo...")

    model = BayesianCoin(prior_a=5, prior_b=2, true_p=0.7, seed=42)
    model.plot_update([10, 30], save_path="bayes_coin_demo.png")
