import numpy as np
import matplotlib.pyplot as plt
from src.linear_regression import LinearRegression

# generate data
X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + 5 + np.random.randn(100)

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

predictions = model.predict(X)

# plot
plt.scatter(X, y)
plt.plot(X, predictions, color='red')
plt.savefig("linear_reg_exp.png")
plt.show()
