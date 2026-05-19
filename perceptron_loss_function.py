from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0,n_classes=2, n_clusters_per_class=1, random_state=41,
                           hypercube=False, class_sep=10)

def perceptron(X, y):
    w1 = w2 = b = 1
    lr = 0.1
    
    for j in range(1000):
        for i in range(X.shape[0]):

            z = w1*X[i][0] + w2*X[i][1] + b

            if z*y[i] < 0:
                w1 += lr*y[i]*X[i][0]
                w2 += lr*y[i]*X[i][1]
                b += lr*y[i]
    
    return w1, w2, b

w1, w2, b = perceptron(X, y)

