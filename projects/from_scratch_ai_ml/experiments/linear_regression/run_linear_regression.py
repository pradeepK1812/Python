from src.linear_regression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# dataset
X = np.random.randn(100, 2)
y = X @ np.array([[3], [2]]) + 1

model = LinearRegression(learning_rate=0.01, verbose=True)
losses = model.fit(X, y)

plt.plot(losses)
plt.title("Loss Curve")
plt.savefig("linear_reg.png")
#plt.show()
