Comparing MLE, Lasso (Regularized MLE), and MAP Estimation
🧩 Overview

This example (compare_mle_lasso_map.py) compares three approaches to estimating parameters
of a simple linear model:

𝑦
=
𝑎
𝑥
+
𝑏
+
𝜀
y=ax+b+ε

where 
𝜀
∼
𝑁
(
0
,
𝜎
2
)
ε∼N(0,σ
2
).

We simulate noisy linear data and compare:

Method	Description	Regularization / Prior
MLE	Maximum Likelihood Estimate — equivalent to Ordinary Least Squares (OLS)	None
Lasso	MLE with an L1 penalty on coefficients (sparse regularization)	Implicit prior: Laplace (
MAP	Maximum A Posteriori Estimate using explicit Gaussian priors	Explicit prior, e.g. 
𝑏
∼
𝑁
(
0.1
,
0.1
2
)
b∼N(0.1,0.1
2
)
🧠 Conceptual Summary
Term	Mathematical Meaning	Intuition
MLE	( \hat{\theta}{MLE} = \arg\max\theta P(D	\theta) )
Lasso	( \hat{\theta} = \arg\max_\theta [\log P(D	\theta) - \lambda
MAP	( \hat{\theta}{MAP} = \arg\max\theta [\log P(D	\theta) + \log P(\theta)] )
⚙️ Implementation Summary

Data Generation

X = np.linspace(0, 10, 50)
y = 2.0 * X + 0.1 + np.random.normal(0, 1, size=len(X))


True slope = 2.0, true intercept = 0.1.

MLE (Ordinary Least Squares)

a_mle, b_mle = np.polyfit(X, y, 1)


Lasso Regression

from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
model.fit(X.reshape(-1, 1), y)


MAP Estimation

Uses a Gaussian prior on intercept (b) and maximizes the posterior:

𝑃
(
𝑎
,
𝑏
∣
𝐷
)
∝
𝑃
(
𝐷
∣
𝑎
,
𝑏
)
×
��
(
𝑏
)
P(a,b∣D)∝P(D∣a,b)×P(b)

Example prior:

prior_b_mean = 0.1
prior_b_std = 0.1

📈 Output

The script plots:

True line (green dashed)

MLE line (blue)

Lasso line (orange)

MAP line (red)

Each fit shows how the choice of regularization or prior affects the slope/intercept estimates.

🚀 Run It

From repo root:

python -m projects.stats_visualizer.examples.compare_mle_lasso_map


This will:

Simulate data

Perform MLE, Lasso, and MAP estimation

Save a comparison plot (e.g., compare_mle_lasso_map.png)

Print coefficients for each method.
