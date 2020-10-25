import numpy as np

class CostFunction:
    def f(self, a_last, y):
        raise NotImplementedError

    def grad(self, a_last, y):
        raise NotImplementedError
        

class SoftMaxCrossEntropy(CostFunction):
    def f(self, a_last, y):
        batch_size = a_last.shape[0]
        cost = -1 / batch_size * (y * np.log(np.clip(a_last, epsilon, 1.0))).sum()
        return cost

    def grad(self, a_last, y):
        return - np.divide(y, np.clip(a_last, epsilon, 1.0))
