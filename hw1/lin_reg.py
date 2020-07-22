import csv
import numpy as np
import sys
from functools import reduce

class linear_regression:
    def __init__(self, time_len = 9, source = None, w = None, no_train = False):
        self.source = source if source else ['CO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'SO2', 'THC', 'WIND_SPEED']
        self.time_len = time_len
        if not no_train:
            self.f = open('train.csv', 'r', encoding='big5')
            self.train_data = self.get_data()
        self.w = w if w else np.array([-1.71365724e-01 , 3.61099201e-02 , 1.47625869e-01 , 8.47787523e-02
                , -5.98896729e-01 , 6.18692751e-01 , -1.43573727e-01 , -3.91451526e-01
                , 2.53378961e-01 , 1.06588990e+00 , -6.41497113e-03 , -5.53459675e-02
                , -1.15211728e-01 , -7.86600959e-02 , 1.56228814e-02 , 7.50676146e-02
                , -1.01260510e-01 , -1.18593192e-01 , 2.38411100e-01 , 3.40400888e-02
                , -2.70954204e-03 , 1.02146541e-01 , 1.47294742e-02 , 8.54699259e-03
                , -6.38823413e-02 , 1.68149795e-02 , 7.10770455e-02 , -3.20819496e-03
                , 1.79882444e-02 , -5.41615300e-03 , -6.37802855e-03 , -1.67069261e-02
                , -6.02128837e-03 , -4.21336838e-03 , -3.21382924e-02 , 6.90567632e-04
                , 8.31511207e-02 , 1.10287309e-03 , 4.47363612e-03 , -4.99648252e-03
                , 1.32558640e-02 , -1.87474660e-02 , 1.53510129e-02 , -1.92162696e-03
                , -2.27973756e-02 , 7.20944800e-02 , -4.15557270e-02 , -2.73513144e-02
                , 2.74490688e-01 , -2.90660409e-01 , -4.74304347e-02 , 5.93011626e-01
                , -6.36669549e-01 , 3.63206550e-03 , 1.00080252e+00 , 2.98091512e-02
                , 5.41847331e-03 , -6.83197818e-02 , -1.10404164e-03 , -4.52701833e-02
                , 4.17191428e-02 , 3.65971858e-03 , -2.40340800e-02 , -7.13609055e-02
                , -2.65895830e-01 , 2.79260554e-01 , -8.50194925e-02 , -4.73535540e-02
                , -3.09649350e-03 , 7.99187385e-02 , -1.29293448e-01 , 1.08820995e-01
                , 1.15092077e-01 , -1.10252727e-01 , -1.28238919e-01 , 2.27677157e-02
                , -1.59515615e-01 , -1.63226684e-03 , 1.17127777e-01 , 2.45756301e-03
                , 5.32443341e-03 , 3.23769090e-01 , -2.05558213e-01 , -5.55007330e-03
                , 1.90410440e-01 , -1.12604605e-01 , -1.92305195e-02 , 1.02524290e-01
                , -1.29030253e-01 , -1.23165226e-01 , -1.06126383e-01])

    def get_data(self):
        num, pos = len(self.source), self.source.index('PM2.5')
        counter, div = 0, 20 * num
        data, source_x, train_y = [[] for _ in range(num)], [[] for _ in range(num)], []

        def collect_month(self):
            nonlocal num, pos, data, source_x, train_y
            for i in range(num):
                for j in range(len(data[i]) - 9):
                    source_x[i].append(data[i][j + (9 - self.time_len): j + 9])
                    if i == pos: train_y.append(float(data[i][j + 9]))
                data[i] = []

        self.f.seek(0)
        for row in csv.reader(self.f):
            if any(s in row for s in self.source):
                data[counter % num] += [x if x != 'NR' else 0 for x in row[3:]]
                counter += 1
                if counter % div == 0: collect_month(self)
        train_x, error = list(zip(*source_x)), []
        for i in range(len(train_x)):
            train_x[i] = [1] + reduce(lambda a, b: a + b, train_x[i]) + [train_y[i]]

        train_x = np.array(train_x, dtype=np.double)
        for i in range(train_x.shape[0]):
            if len(train_x[i][train_x[i] < 0]): error.append(i)
        train_x = np.delete(train_x, error, 0)
        return train_x

    def gradient(self, validate = 0):
        return sum((self.train_data[i][-1] - np.dot(self.w, self.train_data[i][:-1])) * self.train_data[i][:-1] for i in range(self.train_data.shape[0] - validate)) * -2 / (self.train_data.shape[0] - validate)

    def get_result(self, test_file, result_file):
        M = len(self.source)
        output = [['id', 'value']]
        data = list(np.array([number if number != 'NR' else 0 for number in x[2:]], dtype=np.float) for x in csv.reader(open(test_file, 'r')) if any(feature in x for feature in self.source))
        vec = np.ones(1)
        for i in range(len(data)):
            if len(data[i][data[i] < 0]):
                for j in range(len(data[i])):
                    data[i][j] = data[i][j + 1] if j == 0 else data[i][j - 1] if j == len(data[i]) - 1 else (data[i][j - 1] + data[i][j + 1]) // 2
            vec = np.append(vec, data[i])
            if i % M == M - 1:
                output.append([f'id_{i // M}', np.dot(self.w, vec)])
                vec = np.ones(1)
        csv.writer(open(result_file, 'w')).writerows(output)

    def rmsq(self, validate = 0):
        N = self.train_data.shape[0]
        print('On training set:', np.sqrt(sum((self.train_data[i][-1] - np.dot(self.w, self.train_data[i][:-1]))**2 for i in range(N - validate)) / (N - validate)))
        if validate:
            print('On validation set:', np.sqrt(sum((self.train_data[i][-1] - np.dot(self.w, self.train_data[i][:-1]))**2 for i in range(N - validate, N)) / validate))

    def train(self, iter = 5000, validate = 0, verbose = False, seed = 64):
        M = len(self.source)
        if validate:
            np.random.seed(seed)
            np.random.shuffle(self.train_data)
        alpha, beta_1, beta_2, epsilon, v, s, self.w = 0.002, 0.9, 0.999, 1e-7, np.zeros(1 + M * self.time_len), np.zeros(1 + M * self.time_len), np.zeros(1 + M * self.time_len)
        for step in range(1, iter):
            grad = self.gradient(validate)
            v = beta_1 * v + (1. - beta_1) * grad
            s = beta_2 * s + (1. - beta_2) * np.power(grad, 2)
            v_hat = v / (1. - np.power(beta_1, step)) + (1 - beta_1) * grad / (1 - np.power(beta_1, step))
            s_hat = s / (1. - np.power(beta_2, step))
            self.w = self.w - alpha * v_hat / (np.sqrt(s_hat) + epsilon)
            if verbose: self.rmsq(validate)
        self.rmsq(validate)

model = linear_regression(no_train = True)
#model.train()
model.get_result(test_file = sys.argv[1], result_file = sys.argv[2])