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
        self.tol = 1e-3
        self.passes = 10
        self.max_iter = 2000
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
        return np.where(np.dot(x, self.w) + self.b > 0, 1, 0)

    def smo(self):
        np.random.seed(0)

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
        :param y: n
        """
        y = y.reshape(-1, 1)
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
        return s.reshape(-1)


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
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hide_dim, bias=False),
                nn.Sigmoid(),
                nn.Linear(hide_dim, out_dim, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.layers(x).reshape(-1)

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
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).float()
        for _ in range(self.epoch):
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss(y, pred)
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        self.model.eval()
        x = torch.from_numpy(x)
        with torch.no_grad():
            pred = self.model(x)
        return pred.numpy()


class KMeans:
    def __init__(self, n_clusters, metric, steps=10):
        self.n_clusters = n_clusters
        self.steps = steps
        self.metric = metric
        self.tol = 1e-5

    def fit_transform(self, x):
        np.random.seed(0)
        centers = self.init_centers(x)
        clusters = None
        for _ in range(self.steps):
            clusters = self.cluster(centers, x)
            centers_n = self.update_centers(clusters, x)
            if ((centers_n - centers) ** 2).mean() < self.tol:
                break
            centers = centers_n
        return -clusters

    def init_centers(self, x):
        n, m = x.shape[0], self.n_clusters
        return x[np.random.permutation(n)[:m]]

    def cluster(self, centers, x):
        n, m = x.shape[0], self.n_clusters
        clusters = np.zeros((n, m))
        for i in range(n):
            dis = self.metric(centers, x[i])
            idx = dis.argmin()
            clusters[i, idx] = 1
        return clusters

    def update_centers(self, clusters, x):
        m, d = self.n_clusters, x.shape[1]
        centers = np.zeros((m, d))
        for i in range(m):
            centers[i] += clusters[:, i].dot(x)
            centers[i] /= clusters[:, i].sum()
        return centers


def dist_2_index(dist):
    return dist.argmin(axis=1)


class HierarchicalCluster:
    class Node:
        def __init__(self, val, left, right, id_):
            self.value = val
            self.left = left
            self.right = right
            self.id = id_

        def get_leaves(self):
            if self.value is not None:
                return [self.value]
            return self.left.get_leaves() + self.right.get_leaves()

    def __init__(self, n_clusters, metric):
        self.n_clusters = n_clusters
        self.metric = metric
        self.labels_ = None

    def fit(self, x):
        d = self.metric(x, x)
        np.fill_diagonal(d, np.inf)
        n = d.shape[0]

        clusters = {i: HierarchicalCluster.Node(i, None, None, i) for i in range(n)}
        row_index, col_index = -1, -1
        for k in range(n - self.n_clusters):
            print(k)
            min_dis = np.inf
            for i in range(n):
                for j in range(n):
                    if d[i, j] <= min_dis:
                        min_dis = d[i, j]
                        row_index, col_index = i, j
            for i in range(n):
                if i == col_index:
                    continue
                tmp = min(d[col_index, i], d[row_index, i])
                d[col_index, i] = d[i, col_index] = tmp
            for i in range(n):
                d[row_index, i] = d[i, row_index] = np.inf
            minimum = min(col_index, row_index)
            maximum = max(col_index, row_index)
            merged = HierarchicalCluster.Node(None, clusters[minimum], clusters[maximum], n + k)
            clusters[minimum] = merged
            del clusters[maximum]
        self.labels_ = np.zeros(n, dtype=np.int64)
        for i, node in enumerate(clusters.values()):
            for leaf in node.get_leaves():
                self.labels_[leaf] = i


def _test_svm():
    from sklearn import svm
    from sklearn import datasets
    from sklearn.model_selection import train_test_split as ts

    np.random.seed(0)

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    idx = np.argwhere(y < 2)
    idx = [i[0] for i in idx]
    x = x[idx]
    y = y[idx]

    train_x, test_x, train_y, test_y = ts(x, y, test_size=0.3)

    s1 = SVM()
    s2 = svm.SVC(gamma='auto', kernel='linear')

    s1.fit(train_x, train_y)
    s2.fit(train_x, train_y)

    pred1 = s1.predict(test_x).astype(np.int32)
    pred2 = s2.predict(test_x).astype(np.int32)
    print((pred1 == test_y).sum() / pred1.shape[0])
    print((pred2 == test_y).sum() / pred2.shape[0])
    print((pred1 == pred2).sum() / pred1.shape[0])


if __name__ == '__main__':
    _test_svm()
