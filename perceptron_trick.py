from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0,n_classes=2, n_clusters_per_class=1, random_state=41,
                           hypercube=False, class_sep=10)

def step(z):
    return 1 if z >= 0 else 0

def perceptron(X, y):
    X = np.insert(X, 0, 1, axis=1)  # Add bias term
    weights = np.ones(X.shape[1])  # Initialize weights
    lr = 0.1

    for i in range(1000):
        j = np.random.randint(0, 100) # Randomly select a sample
        y_hat = step(np.dot(X[j], weights))  # Predict
        weights = weights + lr*(y[j] - y_hat)*X[j]  # Update weights
    
    return weights[0], weights[1:]

intercept_, coef_ = perceptron(X, y)

print("Intercept:", intercept_)
print("Coefficients:", coef_)

m = -coef_[0] / coef_[1]
b = -intercept_ / coef_[1]