import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate #step size
        self.n_iterations = n_iterations   #training loop control
        self.weights = None  #model paramter theta  
        self.bias = None    # intercept


    def predict(self, X):
        return np.dot(X, self.weights) + self.bias # h(x)=XW+b

    #  cost computation by formula : J(θ)= 1/2m ∑i=1,m(hθ(x(i))−y(i))2
    def _compute_cost(self, y, y_pred):
        m = len(y)
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        return cost


    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            y_pred = self.predict(X)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            self.costs = []

            for i in range(self.n_iterations):
                y_pred = self.predict(X)

                cost = self._compute_cost(y, y_pred)
                self.costs.append(cost)
