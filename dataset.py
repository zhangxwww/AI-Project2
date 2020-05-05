import csv
import numpy as np


class Dataset:
    def __init__(self, task, path, k_folds):
        self.task = task
        self.path = path
        self.k_folds = k_folds

        self.x = None
        self.y = None
        self.num = self._load_data()
        self.sampler = KFoldsSampler(self.num, k_folds)

    def _load_data(self):
        with open(self.path) as f:
            xs, ys = [], []
            reader = csv.DictReader(f)
            for row in reader:
                if self.task == 'classification':
                    x, y = ClassificationDatasetParser.parse_y(row)
                    xs.append(x)
                    ys.append(y)
        self.x = np.stack(xs)
        self.y = np.vstack(ys)
        return self.x.shape[0]

    def get_data(self, fold):
        return self.sampler(self.x, self.y, fold)


class KFoldsSampler:
    def __init__(self, n, k_folds):
        self.order = np.random.permutation(n)
        self.k_folds = k_folds
        self.num_each_fold = n // k_folds

    def __call__(self, x, y, k):
        index = self.order[k * self.num_each_fold: (k + 1) * self.num_each_fold]
        return x[index], y[index]


class ClassificationDatasetParser:
    def __init__(self):
        pass

    @staticmethod
    def parse(row):
        x = []
        x += ClassificationDatasetParser.parse_age(row)
        x += ClassificationDatasetParser.parse_job(row)
        x += ClassificationDatasetParser.parse_marital(row)
        x += ClassificationDatasetParser.parse_education(row)
        x += ClassificationDatasetParser.parse_default(row)
        x += ClassificationDatasetParser.parse_balance(row)
        x += ClassificationDatasetParser.parse_housing(row)
        x += ClassificationDatasetParser.parse_loan(row)
        x += ClassificationDatasetParser.parse_contact(row)
        x += ClassificationDatasetParser.parse_day(row)
        x += ClassificationDatasetParser.parse_month(row)
        x += ClassificationDatasetParser.parse_duration(row)
        x += ClassificationDatasetParser.parse_campaign(row)
        x += ClassificationDatasetParser.parse_pdays(row)
        x += ClassificationDatasetParser.parse_previous(row)
        x += ClassificationDatasetParser.parse_poutcome(row)
        y = ClassificationDatasetParser.parse_y(row)
        return x, y

    @staticmethod
    def parse_age(row):
        return [int(row['age'])]

    @staticmethod
    def parse_job(row):
        job = row['job']
        encodes = {
            'admin.': 0,
            'blue-collar': 1,
            'entrepreneur': 2,
            'housemaid': 4,
            'management': 5,
            'retired': 6,
            'self-employed': 7,
            'services': 8,
            'student': 9,
            'technician': 10,
            'unemployed': 11,
            'unknown': 12
        }
        encode = [0] * len(encodes)
        encode[encodes[job]] = 1
        return encode

    @staticmethod
    def parse_marital(row):
        mar = row['marital']
        encodes = {
            'divorced': 0,
            'married': 1,
            'single': 2
        }
        encode = [0] * len(encodes)
        encode[encodes[mar]] = 1
        return encode

    @staticmethod
    def parse_education(row):
        edu = row['education']
        encodes = {
            'primary': 0,
            'secondary': 1,
            'tertiary': 2,
            'unknown': 3
        }
        encode = [0] * len(encodes)
        encode[encodes[edu]] = 1
        return encode

    @staticmethod
    def parse_default(row):
        default = row['default']
        return [0] if default == 'no' else [1]

    @staticmethod
    def parse_balance(row):
        return [int(row['balance'])]

    @staticmethod
    def parse_housing(row):
        housing = row['housing']
        return [0] if housing == 'no' else [1]

    @staticmethod
    def parse_loan(row):
        loan = row['loan']
        return [0] if loan == 'no' else [1]

    @staticmethod
    def parse_contact(row):
        contact = row['contact']
        encodes = {
            'cellular': 0,
            'telephone': 1,
            'unknown': 2
        }
        encode = [0] * len(encodes)
        encode[encodes[contact]] = 1
        return encode

    @staticmethod
    def parse_day(row):
        return [int(row['day'])]

    @staticmethod
    def parse_month(row):
        month = row['month']
        encodes = {
            'jan': 0,
            'feb': 1,
            'mar': 2,
            'apr': 3,
            'may': 4,
            'jun': 5,
            'jul': 6,
            'aug': 7,
            'sep': 8,
            'oct': 9,
            'nov': 10,
            'dec': 11
        }
        encode = [0] * len(encodes)
        encode[encodes[month]] = 1
        return encode

    @staticmethod
    def parse_duration(row):
        return [int(row['duration'])]

    @staticmethod
    def parse_campaign(row):
        return [int(row['campaign'])]

    @staticmethod
    def parse_pdays(row):
        return [int(row['pdays'])]

    @staticmethod
    def parse_previous(row):
        return [int(row['previous'])]

    @staticmethod
    def parse_poutcome(row):
        poutcome = row['contact']
        encodes = {
            'failure': 0,
            'other': 1,
            'success': 2,
            'unknown': 3
        }
        encode = [0] * len(encodes)
        encode[encodes[poutcome]] = 1
        return encode

    @staticmethod
    def parse_y(row):
        return int(row['y'])

    @staticmethod
    def range(keys):
        pass
