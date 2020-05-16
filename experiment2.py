import numpy as np

from sklearn import cluster
from model import KMeans, HierarchicalCluster, dist_2_index

from dataset import Dataset
from feature import DefaultFeatureSelector, AutoFeatureSelector
from metric import l2, cosine
from score import entropy, purity

from sklearn import manifold

import time
import os

# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set()

N_CLUSTERS = 4


def exp2():
    dataset = prepare_dataset()
    feature_selectors = prepare_feature_selector(dataset.parser)
    x, y = dataset.get_data(None)
    for fs, fs_desc in feature_selectors:
        print(fs_desc)
        model_list = get_model_list()
        x = fs.select(x, y)
        # tsne = TSNE_Plotter(x)
        for metric, metric_desc in prepare_metric():
            print(metric_desc)
            score_list = prepare_score(model_list)
            for model, model_desc in prepare_models(metric):
                print(model_desc)

                start = time.time()
                if is_kmeans(model_desc):
                    clusters = dist_2_index(model.fit_transform(x))
                else:
                    clusters = model.fit(x).labels_
                end = time.time()

                print('{:.3f}s costs'.format(end - start))
                score_list['entropy'][model_desc].append(entropy(y, clusters, N_CLUSTERS))
                score_list['purity'][model_desc].append(purity(y, clusters, N_CLUSTERS))
                save(clusters, model_desc, fs_desc)
                # tsne.plot(y, clusters, model_desc, metric_desc, fs_desc)
            show_results(score_list)


def prepare_dataset():
    return Dataset('cluster', '../data/clustering/Frogs_MFCCs.csv', k_folds=None)


def prepare_feature_selector(parser):
    return [
        (DefaultFeatureSelector(parser), 'All features'),
        (AutoFeatureSelector(parser), 'Features selected according to the statistics')
    ]


def get_model_list():
    return [
        'K-Means 4',
        'K-Means 8',
        'K-Means sklearn 4',
        'K-Means sklearn 8',
        'Hierarchical cluster 4',
        'Hierarchical cluster 8',
        'Hierarchical cluster single sklearn 4',
        'Hierarchical cluster single sklearn 8',
        'Hierarchical cluster ward sklearn 4',
        'Hierarchical cluster ward sklearn 8',
    ]


def prepare_score(model_list):
    en = {m: [] for m in model_list}
    pu = {m: [] for m in model_list}
    return {
        'entropy': en,
        'purity': pu
    }


def prepare_metric():
    return [
        (l2, 'Euclidean distance'),
        # (cosine, 'Cosine distance')
    ]


def prepare_models(metric):
    m = 'l2' if metric == l2 else 'cosine'
    return [
        (KMeans(n_clusters=N_CLUSTERS, metric=metric), 'K-Means 4'),
        (KMeans(n_clusters=N_CLUSTERS * 2, metric=metric), 'K-Means 8'),
        (cluster.KMeans(n_clusters=N_CLUSTERS, random_state=0), 'K-Means sklearn 4'),
        (cluster.KMeans(n_clusters=N_CLUSTERS * 2, random_state=0), 'K-Means sklearn 8'),
        (HierarchicalCluster(n_clusters=N_CLUSTERS, metric=metric), 'Hierarchical cluster 4'),
        (HierarchicalCluster(n_clusters=N_CLUSTERS * 2, metric=metric), 'Hierarchical cluster 8'),
        (cluster.AgglomerativeClustering(linkage='single', n_clusters=N_CLUSTERS),
         'Hierarchical cluster single sklearn 4'),
        (cluster.AgglomerativeClustering(linkage='single', n_clusters=N_CLUSTERS * 2),
         'Hierarchical cluster single sklearn 8'),
        (cluster.AgglomerativeClustering(linkage='ward', n_clusters=N_CLUSTERS),
         'Hierarchical cluster ward sklearn 4'),
        (cluster.AgglomerativeClustering(linkage='ward', n_clusters=N_CLUSTERS * 2),
         'Hierarchical cluster ward sklearn 8')
    ]


def is_kmeans(desc):
    return 'K-Means' in desc


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


def save(clusters, model_desc, fs_desc):
    filename = '{},{}.npy'.format(model_desc, fs_desc)
    np.save(filename, clusters)


'''
class TSNE_Plotter:
    def __init__(self, x):
        tsne = manifold.TSNE()
        self.emb = tsne.fit_transform(x)
        self.color = {
            0: '#1F77B4',
            1: '#FF7F0E',
            2: '#2CA02C',
            3: '#D62728',
            4: '#AEC7E8',
            5: '#FFBB78',
            6: '#C49C94',
            7: '#FF9896',
            8: '#9467BD',
            9: '#8C564B'
        }

    def plot(self, y, clusters, model_desc, _, fs_desc):
        fig = plt.figure(figsize=(12, 5))
        ax0 = fig.add_subplot(121)
        sns.scatterplot(
            x=self.emb[:, 0],
            y=self.emb[:, 1],
            hue=y.astype(np.int32),
            palette=self.color,
            ax=ax0
        )
        title = 'Ground Truth'
        ax0.set_title(title)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax1 = fig.add_subplot(122)
        sns.scatterplot(
            x=self.emb[:, 0],
            y=self.emb[:, 1],
            hue=clusters.astype(np.int32),
            palette=self.color,
            ax=ax1
        )
        title = '{}\n{}'.format(model_desc, fs_desc)
        ax1.set_title(title)
        ax1.set_xticks([])
        ax1.set_yticks([])
        filename = get_filename('{}.png'.format(title.replace('\n', ',')))
        fig.savefig(filename)
'''


def get_filename(filename):
    path = 'results'
    if not os.path.exists(path):
        os.mkdir(path)
    return os.path.join(path, filename)


if __name__ == '__main__':
    exp2()
