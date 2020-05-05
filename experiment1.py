from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from model import SVM, MLP, MLP_Torch

from dataset import Dataset


def prepare_models(d):
    models = [
        (SVM(), 'SVM by myself'),
        (svm.SVC(), 'SVM by sklearn'),
        (MLP(d, 1, d // 2, lr=0.01, epoch=100), 'MLP by myself'),
        (MLP_Torch(d, 1, d // 2, lr=0.01, epoch=100), 'MLP by PyTorch'),
        (tree.DecisionTreeClassifier(), 'Decision Tree by sklearn'),
        (neighbors.KNeighborsClassifier(n_neighbors=5), 'K-nn by sklearn')
    ]
    return models
