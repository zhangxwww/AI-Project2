import numpy as np

THRESHOLD = 0.5


def accuracy(pred, gt):
    a, b, c, d = tp(pred, gt), fn(pred, gt), fp(pred, gt), tn(pred, gt)
    return (a + d) / (a + b + c + d)


def precision(pred, gt):
    a, b, c, d = tp(pred, gt), fn(pred, gt), fp(pred, gt), tn(pred, gt)
    return a / (a + c)


def recall(pred, gt):
    a, b, c, d = tp(pred, gt), fn(pred, gt), fp(pred, gt), tn(pred, gt)
    return a / (a + b)


def f_measure(pred, gt):
    a, b, c, d = tp(pred, gt), fn(pred, gt), fp(pred, gt), tn(pred, gt)
    return 2 * a / (2 * a + b + c)


def tp(pred, gt):
    return ((pred >= THRESHOLD) * gt).sum()


def fn(pred, gt):
    return ((pred < THRESHOLD) * gt).sum()


def fp(pred, gt):
    return ((pred >= THRESHOLD) * (1 - gt)).sum()


def tn(pred, gt):
    return ((pred < THRESHOLD) * (1 - gt)).sum()


def entropy(C, D, k):
    d = [np.argwhere(D == i) for i in range(k)]
    return sum([entropy_i(C, di, k) * di.shape[0] / D.shape[0] for di in d])


def entropy_i(C, di, k):
    n = di.shape[0]
    c_in_di = C[di]
    return -sum([plogp(np.argwhere(c_in_di == i).shape[0] / n) for i in range(k)])


def plogp(x):
    return x * np.log2(x) if x > 0 else 0


def purity(C, D, k):
    d = [np.argwhere(D == i) for i in range(k)]
    return sum([purity_i(C, di, k) * di.shape[0] / D.shape[0] for di in d])


def purity_i(C, di, k):
    n = di.shape[0]
    c_in_di = C[di]
    return max([np.argwhere(c_in_di == i).shape[0] for i in range(k)]) / n
