import numpy as np


def l2(p1, p2):
    """
    :param p1: n * d
    :param p2: m * d
    :return:   n * m
    """
    MAX_SIZE = 25000000
    if p1.ndim == 1:
        p1 = p1.reshape(1, -1)
    if p2.ndim == 1:
        p2 = p2.reshape(1, -1)
    n, m = p1.shape[0], p2.shape[0]
    if n * m < MAX_SIZE:
        p1 = p1[:, np.newaxis, :]
        p2 = p2[np.newaxis, :, :]
        dist = ((p1 - p2) ** 2).sum(2)
    else:
        dist = np.zeros((n, m))
        if n < m:
            k = MAX_SIZE // m
            for i in range(k + 1):
                dist[:, i * k: (i + 1) * k] = l2(p1, p2[i * k: (i + 1) * k])
        else:
            k = MAX_SIZE // n
            for i in range(k + 1):
                dist[i * k: (i + 1) * k, :] = l2(p1[i * k: (i + 1) * k], p2)
    return dist


def cosine(p1, p2):
    """
    :param p1: n * d
    :param p2: m * d
    :return:   n * m
    """
    MAX_SIZE = 25000000
    if p1.ndim == 1:
        p1 = p1.reshape(1, -1)
    if p2.ndim == 1:
        p2 = p2.reshape(1, -1)
    n, m = p1.shape[0], p2.shape[0]
    if n * m < MAX_SIZE:
        p1 = p1[:, np.newaxis, :]
        p2 = p2[np.newaxis, :, :]
        inner = (p1 * p2).sum(2)
        len1 = np.sqrt((p1 * p1).sum(2)).reshape(-1, 1)
        len2 = np.sqrt((p2 * p2).sum(2)).reshape(1, -1)
        dist = inner / len1 / len2
    else:
        dist = np.zeros((n, m))
        if n < m:
            k = MAX_SIZE // m
            for i in range(k + 1):
                dist[:, i * k: (i + 1) * k] = cosine(p1, p2[i * k: (i + 1) * k])
        else:
            k = MAX_SIZE // n
            for i in range(k + 1):
                dist[i * k: (i + 1) * k, :] = cosine(p1[i * k: (i + 1) * k], p2)
    return dist
