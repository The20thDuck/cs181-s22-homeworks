#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

def f(x, tau, data=data):
    y = 0
    for x0, y0 in data:
        if x0 != x:
            k = np.exp(-(x-x0)**2/tau)
            y += k*y0
    return y

def compute_loss(tau):
    # TODO
    loss = 0
    for x, y in data:
        loss += (f(x, tau) - y)**2
    return loss


def f_plot(x, tau, data=data):
    y = 0
    for x0, y0 in data:
        k = np.exp(-(x-x0)**2/tau)
        y += k*y0
    return y

if __name__ == "__main__":
    plt.plot(np.array(data)[:,0], np.array(data)[:,1], 'ro')
    for tau in (0.01, 2, 100):
        print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))
        x = np.arange(0, 12, 0.1)
        plt.plot(x, f_plot(x, tau), label="tau = " + str(tau))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()