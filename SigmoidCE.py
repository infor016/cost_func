import numpy as np

class SigmoidCrossEntropy():
    def f(self, a_last, y):
        batch_size = a_last.shape[0]
        a_last = np.clip(a_last, epsilon, 1.0 - epsilon)
        cost = -1 / batch_size * (y * np.log(np.clip(a_last, epsilon, 1.0))).sum()
        return cost

    def grad(self, a_last, y):
        a_last = np.clip(a_last, epsilon, 1.0 - epsilon)
        return -(np.divide(y, a_last)) - np.divide(1 - y, 1 - a_last)
