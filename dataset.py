import csv
import numpy as np


class Dataset:
    def __init__(self, task, path, k_folds):
        self.task = task
        self.path = path
        self.k_folds = k_folds

        if self.task == 'classification':
            self.parser = ClassificationDatasetParser()
        else:
            self.parser = ClusterDatasetParser()
        self.x = None
        self.y = None
        self.num = self._load_data()
        self.sampler = KFoldsSampler(self.num, k_folds) if k_folds is not None else None

    def _load_data(self):
        with open(self.path) as f:
            xs, ys = [], []
            reader = csv.DictReader(f)
            for row in reader:
                x, y = self.parser.parse(row)
                xs.append(x)
                ys.append(y)
        self.x = np.stack(xs)
        self.y = np.stack(ys)
        return self.x.shape[0]

    def get_data(self, fold):
        if fold is not None:
            return self.sampler(self.x, self.y, fold)
        else:
            return self.x, self.y


class KFoldsSampler:
    def __init__(self, n, k_folds):
        self.order = np.random.permutation(n)
        self.k_folds = k_folds
        self.num_each_fold = n // k_folds

    def __call__(self, x, y, k):
        train_index = np.concatenate([
            self.order[:k * self.num_each_fold],
            self.order[(k + 1) * self.num_each_fold:]
        ])
        test_index = self.order[k * self.num_each_fold: (k + 1) * self.num_each_fold]
        return {
            'train_x': x[train_index].astype('float32'),
            'train_y': y[train_index],
            'test_x': x[test_index].astype('float32'),
            'test_y': y[test_index]
        }


class ClassificationDatasetParser:
    def __init__(self):
        self.all_fields = []
        self.init_fields()
        self.encodes = {}
        self.init_encodes()
        self.dimension = 0
        self.ranges = {}
        self.init_ranges()

    def init_fields(self):
        self.all_fields = [
            'age', 'job', 'marital', 'education', 'default', 'balance',
            'housing', 'loan', 'contact', 'day', 'month', 'duration',
            'campaign', 'pdays', 'previous', 'poutcome'
        ]

    def init_encodes(self):
        self.encodes['job'] = {
            'admin.': 0,
            'blue-collar': 1,
            'entrepreneur': 2,
            'housemaid': 3,
            'management': 4,
            'retired': 5,
            'self-employed': 6,
            'services': 7,
            'student': 8,
            'technician': 9,
            'unemployed': 10,
            'unknown': 11
        }
        self.encodes['marital'] = {
            'divorced': 0,
            'married': 1,
            'single': 2
        }
        self.encodes['education'] = {
            'primary': 0,
            'secondary': 1,
            'tertiary': 2,
            'unknown': 3
        }
        self.encodes['contact'] = {
            'cellular': 0,
            'telephone': 1,
            'unknown': 2
        }
        self.encodes['month'] = {
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
        self.encodes['poutcome'] = {
            'failure': 0,
            'other': 1,
            'success': 2,
            'unknown': 3
        }

    def init_ranges(self):
        last = 0
        for field in self.all_fields:
            l = len(self.encodes[field]) if field in self.encodes.keys() else 1
            self.ranges[field] = list(range(last, last + l))
            last += l
        self.dimension = last

    def parse(self, row):
        x = []
        x += self.parse_age(row)
        x += self.parse_job(row)
        x += self.parse_marital(row)
        x += self.parse_education(row)
        x += self.parse_default(row)
        x += self.parse_balance(row)
        x += self.parse_housing(row)
        x += self.parse_loan(row)
        x += self.parse_contact(row)
        x += self.parse_day(row)
        x += self.parse_month(row)
        x += self.parse_duration(row)
        x += self.parse_campaign(row)
        x += self.parse_pdays(row)
        x += self.parse_previous(row)
        x += self.parse_poutcome(row)
        y = self.parse_y(row)
        return x, y

    @staticmethod
    def parse_age(row):
        return [int(row['age'])]

    def parse_job(self, row):
        job = row['job']
        encodes = self.encodes['job']
        encode = [0] * len(encodes)
        encode[encodes[job]] = 1
        return encode

    def parse_marital(self, row):
        mar = row['marital']
        encodes = self.encodes['marital']
        encode = [0] * len(encodes)
        encode[encodes[mar]] = 1
        return encode

    def parse_education(self, row):
        edu = row['education']
        encodes = self.encodes['education']
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

    def parse_contact(self, row):
        contact = row['contact']
        encodes = self.encodes['contact']
        encode = [0] * len(encodes)
        encode[encodes[contact]] = 1
        return encode

    @staticmethod
    def parse_day(row):
        return [int(row['day'])]

    def parse_month(self, row):
        month = row['month']
        encodes = self.encodes['month']
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

    def parse_poutcome(self, row):
        poutcome = row['poutcome']
        encodes = self.encodes['poutcome']
        encode = [0] * len(encodes)
        encode[encodes[poutcome]] = 1
        return encode

    @staticmethod
    def parse_y(row):
        return int(row['y'])

    def range(self, keys):
        r = []
        for key in keys:
            assert key in self.all_fields
            r += self.ranges[key][:]
        return r


class ClusterDatasetParser:
    def __init__(self):
        self.y_encodes = self.init_y_encodes()
        self.dimension = 22

    @staticmethod
    def init_y_encodes():
        return {
            'Leptodactylidae': 0,
            'Dendrobatidae': 1,
            'Hylidae': 2,
            'Bufonidae': 3
        }

    def parse(self, row):
        x = [row['MFCCs_{:>2}'.format(i)] for i in range(1, 23)]
        x = np.array(list(map(float, x)))
        y = self.y_encodes[row['Family']]
        return x, y
