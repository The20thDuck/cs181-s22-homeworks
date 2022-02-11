#####################
# CS 181, Spring 2022
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# set up data
data = [(0., 0.),
        (1., 0.5),
        (2., 1),
        (3., 2),
        (4., 1),
        (6., 1.5),
        (8., 0.5)]

x_train = np.array([d[0] for d in data])[::-1]
y_train = np.array([d[1] for d in data])[::-1]

x_test = np.arange(0, 12, .1)

print("y is:")
print(y_train)



def predict_knn(k=1, tau=1):
    """Returns predictions for the values in x_test, using KNN predictor with the specified k."""
    # TODO: your code here
    # K(x, x') is monotonically increasing, so we can just use abs instead
    k_neighbors = np.argsort(np.abs(np.expx_train - x_test.reshape((-1, 1))))[:,:k]
    return np.mean(y_train[k_neighbors], axis=1)

def plot_knn_preds():
    plt.xlim([0, 12])
    plt.ylim([0,3])
    
    
    plt.scatter(x_train, y_train, label = "training data", color = 'black')

    for k in (1, 3, len(x_train)-1):
        y_test = predict_knn(k=k)
        plt.plot(x_test, y_test, label = "predictions using k = " + str(k))

    plt.legend()
    plt.title("KNN Predictions")
    plt.savefig("knn_predictions.png")
    plt.show()

if __name__ == "__main__":
    plot_knn_preds()