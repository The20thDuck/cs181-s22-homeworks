import numpy as np
from scipy.stats import multivariate_normal as mvn
import pandas as pd
# from T2_P1 import N  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        # MLE
        self.K = max(y) + 1
        n = X.shape[0]
        d = X.shape[1]
        one_hot = np.zeros((n, self.K))
        one_hot[np.arange(n), y] = 1 # (n, K)

        self.pi = np.sum(one_hot, axis = 0)/n # (K, )
        self.mu = np.dot(one_hot.T, X)/np.sum(one_hot.T, axis=1, keepdims=True) # (K, n)
        if self.is_shared_covariance:
            self.sigma = np.zeros((d, d))
            for i in range(n):
                x_mu = X[i,:]-self.mu[y[i],:]
                x_mu = x_mu.reshape((-1, 1))
                self.sigma += np.dot(x_mu, x_mu.T)
            self.sigma /= n 
            self.mvns = []
            for k in range(self.K):
                self.mvns.append(mvn(self.mu[k,:], self.sigma))
        else:
            self.sigmas = [np.zeros((d, d)) for k in range(self.K)]
            for i in range(n):
                x_mu = X[i,:]-self.mu[y[i],:]
                x_mu = x_mu.reshape((-1, 1))
                self.sigmas[y[i]] += np.dot(x_mu, x_mu.T)
            for k in range(self.K):
                self.sigmas[k] /= np.sum(one_hot[:, k])
            self.mvns = []
            for k in range(self.K):
                self.mvns.append(mvn(self.mu[k,:], self.sigmas[k]))

    def __predict_probs(self, X_pred):
        vals = np.stack(tuple(self.mvns[k].pdf(X_pred) for k in range(self.K)), axis=-1) * self.pi.reshape((1, -1))
        return vals/np.sum(vals, axis=1, keepdims=True)

    # TODO: Implement this method!
    def predict(self, X_pred, get_probs=False):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        probs = self.__predict_probs(X_pred)
        pred = np.argmax(probs, axis=1)
        if get_probs:
            print(probs)
        return pred

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        y_hat = self.__predict_probs(X)
        n=X.shape[0]

        one_hot = np.zeros((n, self.K))
        one_hot[np.arange(n), y] = 1 # (n, K)
        return -np.sum(np.log(np.max(y_hat * one_hot, axis=1)))

if __name__ == "__main__":
    eta = 0.1 # Learning rate
    lam = 0.1 # Lambda for regularization
    star_labels = {
        'Dwarf': 0,       # also corresponds to 'red' in the graphs
        'Giant': 1,       # also corresponds to 'blue' in the graphs
        'Supergiant': 2   # also corresponds to 'green' in the graphs
    }

    df = pd.read_csv('data/hr.csv')
    X = df[['Magnitude', 'Temperature']].values
    y = np.array([star_labels[x] for x in df['Type']])

    lr = GaussianGenerativeModel(is_shared_covariance=True)
    lr.fit(X, y)
    print(lr.predict(X))