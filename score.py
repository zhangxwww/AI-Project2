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
    p, r = precision(pred, gt), recall(pred, gt)
    return 2 * r * p / (r + p)


def tp(pred, gt):
    return ((pred >= THRESHOLD) * gt).sum()


def fn(pred, gt):
    return ((pred < THRESHOLD) * gt).sum()


def fp(pred, gt):
    return ((pred >= THRESHOLD) * (1 - gt)).sum()


def tn(pred, gt):
    return ((pred < THRESHOLD) * (1 - gt)).sum()
