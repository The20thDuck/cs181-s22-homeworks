import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax

# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.runs = 200000
        self.num_iters = []
        self.log_ls = []

    # Just to show how to make 'private' methods
    def __basis1(self, x):
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    def __predict_probs(self, X_pred):
        transformed_X = self.__basis1(X_pred)
        z = np.dot(transformed_X, self.W)
        pred = softmax(z, axis=1)
        return pred

    # TODO: Implement this method!
    def fit(self, X, y):
        n = X.shape[0]
        num_classes = max(y)+1
        self.W = np.random.rand(X.shape[1] + 1, num_classes)
        one_hot = np.zeros((n, num_classes))
        one_hot[np.arange(n),y] = 1
        for i in range(self.runs):
            y_hat = self.__predict_probs(X)
            grad = (np.dot(self.__basis1(X).T, y_hat - one_hot) + self.lam * self.W)/n
            if (i % 100 == 0):
                log_l = np.sum(np.log(np.max(y_hat * one_hot, axis=1)))
                self.num_iters.append(i)
                self.log_ls.append(-log_l)
            self.W -= self.eta * grad
        return None

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
    def visualize_loss(self, output_file, show_charts=False):
        # plot with -log likelihood loss vs num iters
        plt.title(f"-Log Likelihood vs #iters, eta={self.eta} lam={self.lam}")
        plt.xlabel('#iters')
        plt.ylabel('-Log L')
        plt.plot(self.num_iters, self.log_ls, label=f'lam={self.lam}')
        plt.savefig(output_file)
        if show_charts:
            plt.show()
        

if __name__ == "__main__":
    vals = [.05, .01, .001]
    star_labels = {
        'Dwarf': 0,       # also corresponds to 'red' in the graphs
        'Giant': 1,       # also corresponds to 'blue' in the graphs
        'Supergiant': 2   # also corresponds to 'green' in the graphs
    }

    df = pd.read_csv('data/hr.csv')
    X = df[['Magnitude', 'Temperature']].values
    y = np.array([star_labels[x] for x in df['Type']])
    for eta in vals:
        for lam in vals:
            lr = LogisticRegression(eta=eta, lam=lam)
            lr.fit(X, y)
            lr.visualize_loss(f'logistic_regression_loss', show_charts=False)
        plt.legend()
        plt.savefig(f'logistic_regression_loss eta={eta}.png')
        plt.show()
