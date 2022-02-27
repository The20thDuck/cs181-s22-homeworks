import numpy as np
import pandas as pd
from scipy.stats import mode

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __dist(self, x1, x2):
        diff = x1-x2
        return (diff[0]/3)**2+ (diff[1])**2

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for x in X_pred:
            distances = np.sum(((x - self.X)*np.array([[1/3, 1]]))**2, axis=1)
            preds.append(mode(self.y[np.argsort(distances)[:self.K]])[0])
        return np.array(preds)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y

if __name__ == "__main__":
    eta = 0.1 # Learning rate
    lam = 0.1 # Lambda for regularization
    star_labels = {
        'Dwarf': 0,       # also corresponds to 'red' in the graphs
        'Giant': 1,       # also corresponds to 'blue' in the graphs
        'Supergiant': 2   # also corresponds to 'green' in the graphs
    }
    print(star_labels)
    df = pd.read_csv('data/hr.csv')
    X = df[['Magnitude', 'Temperature']].values
    y = np.array([star_labels[x] for x in df['Type']])

    lr = KNNModel(1)
    lr.fit(X, y)
    print(lr.predict(X))