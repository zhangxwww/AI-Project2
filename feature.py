from sklearn import feature_selection
from sklearn import preprocessing


class DefaultFeatureSelector:
    """
    Use all features
    """

    def __init__(self, parser):
        self.dimension = parser.dimension

    @staticmethod
    def select(x, _):
        return preprocessing.MinMaxScaler().fit_transform(x)


class ManualFeatureSelector:
    """
    Extract features by hand
    """

    def __init__(self, parser):
        self.feature_index = parser.range([
            'job', 'default', 'balance', 'housing', 'loan',
            'duration', 'campaign', 'pdays', 'previous', 'poutcome'
        ])
        self.dimension = len(self.feature_index)

    def select(self, x, _):
        x = x[:, self.feature_index]
        x = preprocessing.MinMaxScaler().fit_transform(x)
        return x


class AutoFeatureSelector:
    """
    Extract features by \Chi^2 statistics
    """

    def __init__(self, parser):
        self.dimension = parser.dimension // 2
        self.selector = feature_selection.SelectKBest(
            score_func=feature_selection.chi2,
            k=self.dimension
        )

    def select(self, x, y):
        x = preprocessing.MinMaxScaler().fit_transform(x)
        if y is not None:
            return self.selector.fit_transform(x, y)
        else:
            return self.selector.transform(x)
