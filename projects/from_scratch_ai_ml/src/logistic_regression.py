import numpy as np

class LogisticRegression:

    #Initialization of logistic regression class 
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None


    #sigmoid function to covert the prediction data  to probability

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    #probability prediction using the sigmoid conversion function
    def predict_proba(self, X):
       z = np.dot(X, self.weights) + self.bias
       return self._sigmoid(z)

#computation of loss using negative log of likelihood
    def _compute_loss(self, y, y_pred):
       eps = 1e-8
       loss = -np.mean(
            y * np.log(y_pred + eps) +
            (1 - y) * np.log(1 - y_pred + eps)
            )

       return loss

#Training loop 
    def fit(self, X, y):
        n_samples, n_features = X.shape

        if y.ndim == 1:
           y = y.reshape(-1, 1)

        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        self.losses = []

        for i in range(self.n_iterations):
        # forward
            y_pred = self.predict_proba(X)

        # loss
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)

        # gradients
            dz = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)

        # update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        # Optional logging
            if i % 100 == 0:
               print(f"Iter {i:04d} | Loss: {loss:.6f}")

        return self.losses

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)


