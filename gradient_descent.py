import math

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    n_samples = len(X)
    n_features = len(X[0]) if n_samples > 0 else 0
    weights = [0.0] * n_features
    bias = 0.0

    for _ in range(epochs):
        y_pred = [sum(X[i][j] * weights[j] for j in range(n_features)) + bias for i in range(n_samples)]
        dw = [0.0] * n_features
        for j in range(n_features):
            dw[j] = (1/n_samples) * sum((y_pred[i] - y[i]) * X[i][j] for i in range(n_samples))
        db = (1/n_samples) * sum(y_pred[i] - y[i] for i in range(n_samples))
        weights = [weights[j] - learning_rate * dw[j] for j in range(n_features)]
        bias -= learning_rate * db

    return weights, bias
