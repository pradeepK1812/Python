import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -----------------------
# Generate data
# -----------------------
np.random.seed(0)
N = 1000

X = np.random.randn(N, 1)
true_w = 2.0
true_b = -0.5

z = true_w * X + true_b
y = (z > 0).astype(int)
y = y.flatten()
# -----------------------
# Train / test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------
# Normalize (standard practice)
# -----------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------
# Model
# -----------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------
# Evaluate
# -----------------------
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("w:", model.coef_[0][0])
print("b:", model.intercept_[0])
print("train accuracy:", train_acc)
print("test accuracy:", test_acc)

