from sklearn import feature_selection

from dataset import ClassificationDatasetParser


class DefaultFeatureSelector:
    """
    Use all features
    """

    def __init__(self, parser):
        self.dimension = parser.dimension

    @staticmethod
    def select(x, _):
        return x


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
        return x[:, self.feature_index]


class AutoFeatureSelector:
    """
    Extract features by \Chi^2 statistics
    """

    def __init__(self, parser):
        self.selector = feature_selection.SelectPercentile(
            score_func=feature_selection.chi2,
            percentile=50
        )
        self.dimension = parser.dimension // 2

    def select(self, x, y):
        return self.selector.fit_transform(x, y)
