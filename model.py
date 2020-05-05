import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SVM:
    def __init__(self):
        self.x = None
        self.y = None
        self.penalty = 1
        self.tol = 1e-4
        self.passes = 10
        self.max_iter = 1000
        self.k = None
        self.alpha = None
        self.w = None
        self.b = None

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.k = np.dot(self.x, self.x.T)
        self.alpha = np.random.normal(0, 1, x.shape[0])
        self.b = self.smo()
        self.w = np.dot(self.alpha * self.y, self.x)

    def predict(self, x):
        return np.sign(np.dot(x, self.w) + self.b)

    def smo(self):
        n = self.x.shape[0]
        it = 0
        p = 0
        b = np.random.normal(0, 1)

        yk = self.y * self.k

        while p < self.passes and it < self.max_iter:
            alpha_changed = 0
            for i in range(n):
                ei = np.dot(self.alpha, yk[i]) + b - self.y[i]
                yei = self.y[i] * ei
                if (yei < -self.tol and self.alpha[i] < self.penalty) or (yei > self.tol and self.alpha[i] > 0):
                    j = i
                    while j == i:
                        j = random.randint(0, n - 1)
                    ej = np.dot(self.alpha, yk[j]) + b - self.y[j]

                    ai = self.alpha[i]
                    aj = self.alpha[j]
                    if self.y[i] == self.y[j]:
                        l = max(0, ai + aj - self.penalty)
                        h = min(ai + aj, self.penalty)
                    else:
                        l = max(0, aj - ai)
                        h = min(aj - ai + self.penalty, self.penalty)

                    if abs(l - h) < 1e-4:
                        continue

                    eta = 2 * self.k[i, j] - self.k[i, i] - self.k[j, j]
                    if eta >= 0:
                        continue

                    newaj = aj - self.y[j] * (ei - ej) / eta
                    newaj = min(newaj, h)
                    newaj = max(newaj, l)
                    if abs(aj - newaj) < 1e-4:
                        continue

                    self.alpha[j] = newaj
                    newai = ai + self.y[i] * self.y[j] * (aj - newaj)
                    self.alpha[i] = newai

                    b1 = b - ei - self.y[i] * (newai - ai) * self.k[i, i] - self.y[j] * (newaj - aj) * self.k[i, j]
                    b2 = b - ej - self.y[i] * (newai - ai) * self.k[i, j] - self.y[j] * (newaj - aj) * self.k[j, j]

                    if 0 < newai < self.penalty:
                        b = b1
                    elif 0 < newaj < self.penalty:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

                    alpha_changed += 1

            it += 1
            if alpha_changed == 0:
                p += 1
            else:
                p = 0

        return b


class MLP:
    def __init__(self, in_dim, out_dim, hide_dim, lr, epoch):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hide_dim = hide_dim
        self.lr = lr
        self.epoch = epoch

        self.w1 = np.random.normal(0, pow(hide_dim, -0.5), (in_dim, hide_dim))
        self.w2 = np.random.normal(0, pow(out_dim, -0.5), (hide_dim, out_dim))

    def fit(self, x, y):
        """
        X1 = XW1
        H = sigmoid(X1)
        X2 = HW2
        S = sigmoid(X2)

        :param x: n * d
        :param y: n * 1
        """
        for _ in range(self.epoch):
            w1, w2 = self.w1, self.w2
            x1 = x.dot(w1)
            h = sigmoid(x1)
            x2 = h.dot(w2)
            s = sigmoid(x2)

            tmp = (s - y) * d_sigmoid(x2)
            delta1 = x.T.dot(tmp.dot(w2.T) * d_sigmoid(x1))
            delta2 = h.T.dot(tmp)

            self.w1 -= self.lr * delta1
            self.w2 -= self.lr * delta2

    def predict(self, x):
        w1, w2 = self.w1, self.w2
        x1 = x.dot(w1)
        h = sigmoid(x1)
        x2 = h.dot(w2)
        s = sigmoid(x2)
        return s


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    """
    sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    """
    return sigmoid(x) * (1 - sigmoid(x))


class MLP_Torch:
    class Module(nn.Module):
        def __init__(self, in_dim, out_dim, hide_dim):
            super().__init__()
            self.layers = nn.Sequential([
                nn.Linear(in_dim, hide_dim),
                nn.Sigmoid(),
                nn.Linear(hide_dim, out_dim),
                nn.Sigmoid()
            ])

        def forward(self, x):
            return self.layers(x)

    def __init__(self, in_dim, out_dim, hide_dim, lr, epoch):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hide_dim = hide_dim
        self.lr = lr
        self.epoch = epoch

        self.model = MLP_Torch.Module(in_dim, out_dim, hide_dim)
        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    def fit(self, x, y):
        self.model.train()
        for _ in range(self.epoch):
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss(y, pred)
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)
        return pred
