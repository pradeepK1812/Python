import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)

N = 1000
X = 2 * np.random.rand(N, 1)
y = 3.5 * X + 1.2 + np.random.randn(N, 1) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

print("w:", model.coef_[0][0])
print("b:", model.intercept_[0])
print("train score:", model.score(X_train, y_train))
print("test score:", model.score(X_test, y_test))

