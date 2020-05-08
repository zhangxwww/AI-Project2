import numpy as np


def l2(p1, p2):
    """
    :param p1: n * d
    :param p2: m * d
    :return:   n * m
    """
    if p1.ndim == 1:
        p1 = p1.reshape(1, -1)
    if p2.ndim == 1:
        p2 = p2.reshape(1, -1)
    p1 = p1[:, np.newaxis, :]
    p2 = p2[np.newaxis, :, :]
    dist = ((p1 - p2) ** 2).sum(2)
    return dist


def cosine(p1, p2):
    """
    :param p1: n * d
    :param p2: m * d
    :return:   n * m
    """
    if p1.ndim == 1:
        p1 = p1.reshape(1, -1)
    if p2.ndim == 1:
        p2 = p2.reshape(1, -1)
    p1 = p1[:, np.newaxis, :]
    p2 = p2[np.newaxis, :, :]
    inner = (p1 * p2).sum(2)
    len1 = (p1 * p1).sum(1).sqrt().reshape(-1, 1)
    len2 = (p2 * p2).sum(1).sqrt().reshape(1, -1)
    return inner / len1 / len2
