import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    """
    Calculates the Euclidean distance between two vectors.
    """
    return np.sqrt(np.sum((x1 - x2)**2))

class KNeighborsClassifier:
    """
    K-Nearest Neighbors classifier from scratch.
    """
    def __init__(self, k=3):
        """
        Initializes the KNN classifier.
        Args:
            k (int): The number of nearest neighbors to consider.
        """
        self.k = k

    def fit(self, X, y):
        """
        Fits the model using the training data.
        Args:
            X (array-like): Training data, shape (n_samples, n_features).
            y (array-like): Target values, shape (n_samples,).
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """
        Predicts the class labels for the provided data.
        Args:
            X (array-like): Test data, shape (n_samples, n_features).
        Returns:
            list: Predicted class labels for each data point.
        """
        X = np.array(X)
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        """
        Helper function to predict a single data point.
        Args:
            x (array): A single sample, shape (n_features,).
        Returns:
            The predicted class label.
        """
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

if __name__ == '__main__':
    # Example Usage
    # Training data
    X_train = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]]
    y_train = ['A', 'A', 'A', 'B', 'B', 'B']

    # Create a KNN classifier with k=3
    classifier = KNeighborsClassifier(k=3)

    # Fit the model
    classifier.fit(X_train, y_train)

    # Test data
    X_test = [[5, 4], [2, 2]]

    # Get predictions
    predictions = classifier.predict(X_test)

    print(f"Predictions for {X_test}: {predictions}")

    # Example with different k
    classifier_k5 = KNeighborsClassifier(k=5)
    classifier_k5.fit(X_train, y_train)
    predictions_k5 = classifier_k5.predict(X_test)
    print(f"Predictions for {X_test} with k=5: {predictions_k5}")