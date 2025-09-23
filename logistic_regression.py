import math

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0.0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0]) if n_samples > 0 else 0
        self.weights = [0.0] * n_features

        for _ in range(self.epochs):
            linear_model = [sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias for i in range(n_samples)]
            y_pred = [self.sigmoid(linear_model[i]) for i in range(n_samples)]
            dw = [0.0] * n_features
            for j in range(n_features):
                dw[j] = (1/n_samples) * sum((y_pred[i] - y[i]) * X[i][j] for i in range(n_samples))
            db = (1/n_samples) * sum(y_pred[i] - y[i] for i in range(n_samples))
            self.weights = [self.weights[j] - self.learning_rate * dw[j] for j in range(n_features)]
            self.bias -= self.learning_rate * db

    def predict(self, X):
        n_samples = len(X)
        n_features = len(X[0]) if n_samples > 0 else 0
        linear_model = [sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias for i in range(n_samples)]
        y_pred = [self.sigmoid(linear_model[i]) for i in range(n_samples)]
        return [1 if i > 0.5 else 0 for i in y_pred]