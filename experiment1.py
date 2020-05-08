import numpy as np

from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from model import SVM, MLP, MLP_Torch

from dataset import Dataset

from feature import DefaultFeatureSelector, ManualFeatureSelector, AutoFeatureSelector

from score import accuracy, precision, recall, f_measure

from utils import timing
import time

N_FOLDS = 5


def exp1():
    dataset = prepare_dataset(N_FOLDS)
    feature_selectors = prepare_feature_selector(dataset.parser)
    for fs, fs_desc in feature_selectors:
        print(fs_desc)
        model_list = get_model_list()
        score_list = prepare_score(model_list)
        for fold in range(N_FOLDS):
            data = dataset.get_data(fold)
            for model, model_desc in prepare_models(fs.dimension):
                print(model_desc)
                train_x = fs.select(data['train_x'], data['train_y'])
                test_x = fs.select(data['test_x'], None)
                start = time.time()
                model.fit(train_x, data['train_y'])
                pred = model.predict(test_x)
                end = time.time()
                print('{:.3f}s costs'.format(end - start))
                gt = data['test_y']
                score_list['accuracy'][model_desc].append(accuracy(pred, gt))
                score_list['precision'][model_desc].append(precision(pred, gt))
                score_list['recall'][model_desc].append(recall(pred, gt))
                score_list['f_measure'][model_desc].append(f_measure(pred, gt))
        show_results(score_list)


def prepare_models(d):
    return [
        (SVM(), 'SVM'),
        (svm.SVC(gamma='auto'), 'SVM_sklearn'),
        (MLP(d, 1, d // 2, lr=0.01, epoch=100), 'MLP'),
        (MLP_Torch(d, 1, d // 2, lr=0.01, epoch=100), 'MLP_Torch'),
        (tree.DecisionTreeClassifier(), 'Tree_sklearn'),
        (neighbors.KNeighborsClassifier(n_neighbors=1), 'K-nn')
    ]


def get_model_list():
    return ['SVM', 'SVM_sklearn', 'MLP', 'MLP_Torch', 'Tree_sklearn', 'K-nn']


def prepare_dataset(folds):
    return Dataset('classification', '../data/classification/train_set.csv', k_folds=folds)


def prepare_feature_selector(parser):
    return [
        (DefaultFeatureSelector(parser), 'All features'),
        (ManualFeatureSelector(parser), 'Features selected by hand'),
        # (AutoFeatureSelector(parser), 'Features selected according to the statistics')
    ]


def prepare_score(model_list):
    acc = {m: [] for m in model_list}
    pre = {m: [] for m in model_list}
    re = {m: [] for m in model_list}
    f = {m: [] for m in model_list}
    return {
        'accuracy': acc,
        'precision': pre,
        'recall': re,
        'f_measure': f
    }


def show_results(score_list):
    models = get_model_list()
    print(' ' * 10, end='')
    for m in models:
        print('{:<12}'.format(m), end=' ')
    print()
    for k, v in score_list.items():
        print('{:<9}'.format(k), end=' ')
        for m in models:
            mean = np.mean(v[m])
            mean = '{:.3f}'.format(mean)
            print('{:<12}'.format(mean), end=' ')
        print()


if __name__ == '__main__':
    exp1()
