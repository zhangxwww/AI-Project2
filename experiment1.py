from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from model import SVM, MLP, MLP_Torch

from dataset import Dataset

from feature import DefaultFeatureSelector, ManualFeatureSelector, AutoFeatureSelector

from score import accuracy, precision, recall, f_measure

N_FOLDS = 5


def exp1():
    dataset = prepare_dataset(N_FOLDS)
    feature_selectors = prepare_feature_selector(dataset.parser)
    for fs, fs_desc in feature_selectors:
        print(fs_desc)
        models, model_list = prepare_models(fs.dimension)
        acc_list, pre_list, recall_list, f_list = prepare_score(model_list)
        for fold in range(N_FOLDS):
            data = dataset.get_data(fold)
            for model, model_desc in models:
                print(model_desc)
                model.fit(data['train_x'], data['train_y'])
                pred = model.predict(data['test_x'])
                gt = data['test_y']
                acc_list[model_desc].append(accuracy(pred, gt))
                pre_list[model_desc].append(precision(pred, gt))
                recall_list[model_desc].append(recall(pred, gt))
                f_list[model_desc].append(f_measure(pred, gt))
        print(acc_list)
        print(pre_list)
        print(recall_list)
        print(f_list)


def prepare_models(d):
    return [
               (SVM(), 'SVM'),
               (svm.SVC(), 'SVM_sklearn'),
               (MLP(d, 1, d // 2, lr=0.01, epoch=100), 'MLP'),
               (MLP_Torch(d, 1, d // 2, lr=0.01, epoch=100), 'MLP_Torch'),
               (tree.DecisionTreeClassifier(), 'DecisionTree_sklearn'),
               (neighbors.KNeighborsClassifier(n_neighbors=5), 'K-nn')
           ], [
               'SVM', 'SVM_sklearn', 'MLP', 'MLP_Torch', 'DecisionTree_sklearn', 'K-nn'
           ]


def prepare_dataset(folds):
    return Dataset('classification', '../data/classification/train_set.csv', k_folds=folds)


def prepare_feature_selector(parser):
    return [
        (DefaultFeatureSelector(parser), 'All features'),
        (ManualFeatureSelector(parser), 'Features selected by hand'),
        (AutoFeatureSelector(parser), 'Features selected according to the statistics')
    ]


def prepare_score(model_list):
    acc = {m: [] for m in model_list}
    pre = {m: [] for m in model_list}
    re = {m: [] for m in model_list}
    f = {m: [] for m in model_list}
    return acc, pre, re, f
