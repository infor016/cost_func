def softmax_cross_entropy(y_hat, y, eps=1e-20) -> float:
    """
    :param y_hat - 2D one-hot prediction tensor with shape (n, k)
    :param y - 2D one-hot ground truth labels tensor with shape (n, k)
    ----------------------------------------------------------------------------
    n - number of examples in batch
    k - number of classes
    """
    n = y_hat.shape[0]
    return - np.sum(y * np.log(np.clip(y_hat, eps, 1.))) / n
