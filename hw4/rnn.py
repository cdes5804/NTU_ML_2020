import torch
import numpy as np
import csv
import torch.optim as optim
import torch.nn as nn
from gensim.models import Word2Vec
from torch.utils.data import DataLoader, Dataset
import sys


class RNN:

    def __init__(self):
        self.train_X, self.train_Y, self.unlabel, self.test_X, self.validate_X, self.validate_Y = [], [], [], [], [], []
        self.w2v = None
        self.label_path, self.unlabel_path, self.test_path, self.result_path = '', '', '', ''

    def get_data(self, is_train = True, label = True):
        X, Y = [], []
        if is_train and label:
            f = open(self.label_path, 'r')
            for line in f:
                data = line.strip().split(' ')
                X.append(data[2:])
                Y.append(int(data[0]))
            return X, Y
        elif is_train and not label:
            f = open(self.unlabel_path, 'r')
            return [line.strip().split(' ') for line in f]
        else:
            f = open(self.test_path, 'r')
            return [''.join(line.strip().split(',')[1:]).split(' ') for line in f][1:]
    
    def get_train_data(self, label = True):
        if label: 
            self.train_X, self.train_Y = self.get_data()
            self.train_Y = np.array(self.train_Y)
        else: self.unlabel = self.get_data(label = False)
    
    def get_test_data(self):
        self.test_X = self.get_data(is_train = False, label = False)
    
    def evaluation(self, pred, label):
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return torch.sum(torch.eq(pred, label)).item()
    
    def save(self, path = './model'):
        torch.save(self.Rnn.state_dict(), path)
        print(f'model saved {path}')
    
    def load(self, path = './model'):
        self.Rnn.load_state_dict(torch.load(path))
        print(f'model loaded {path}')

    def train(self, batch_size = 128, epoch = 100, save_path = './model'):
        print(f'training with {save_path}')
        loss_function = nn.BCELoss()
        train_loader = self.create_data_set(self.train_X, self.train_Y, shuffle = True, batch_size = batch_size)
        if len(self.validate_X):
            validate_loader = self.create_data_set(self.validate_X, self.validate_Y, shuffle = True, batch_size = batch_size)
        best_acc = 0
        optimizer = optim.Adam(self.Rnn.parameters(), lr = 0.001)
        train_acc, validate_acc = 0, 0
        for e in range(epoch):
            self.Rnn.train()
            loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                pred = self.Rnn(x.cuda())
                pred = pred.squeeze()
                Loss = loss_function(pred, y.float().cuda())
                Loss.backward()
                optimizer.step()
                train_acc += self.evaluation(pred, y.float().cuda()) / batch_size
                loss += Loss.item()

            train_acc /= len(train_loader)
            print(f'epoch: {e}\ntrain acc: {train_acc}, train loss: {loss / len(train_loader)}')

            if len(self.validate_X):
                self.Rnn.eval()
                with torch.no_grad():
                    loss = 0
                    for x, y in validate_loader:
                        pred = self.Rnn(x.cuda())
                        pred = pred.squeeze()
                        Loss = loss_function(pred, y.float().cuda())
                        validate_acc += self.evaluation(pred, y.float().cuda()) / batch_size
                        loss += Loss.item()
                validate_acc /= len(validate_loader)
                print(f'val acc: {validate_acc}, val loss: {loss / validate_loader.__len__()}')
                if validate_acc > best_acc:
                    best_acc = validate_acc
                    self.save(save_path)

    def test(self, batch_size = 128, result_path = './result.csv', model_path = ['./model']):
        output = [['id', 'label']]
        self.test_X = self.word2vec(self.test_X)
        test_loader = self.create_data_set(self.test_X, shuffle = False, batch_size = batch_size)
        res = [[0, 0] for _ in range(len(self.test_X))]
        for model in model_path:
            self.init(train_embedding = False, embedding_dim = 200, hidden_dim = 100, dropout = 0.5, layers=2)
            self.load(path = model)
            self.Rnn.eval()
            counter = 0
            with torch.no_grad():
                for x in test_loader:
                    pred = self.Rnn(x.cuda())
                    pred = pred.squeeze()
                    pred[pred >= 0.5] = 1
                    pred[pred < 0.5] = 0
                    pred = pred.int().tolist()
                    for y in pred:
                        res[counter][y] += 1
                        counter += 1
            print(f'finished testing {model}')
        counter = 0
        for results in res:
            if results[0] >= results[1]:
                output.append([counter, 0])
            else:
                output.append([counter, 1])
            counter += 1
        csv.writer(open(result_path, 'w')).writerows(output)
        print('test complete')

    def train_w2v(self, x, save = True, path = './w2v_model', size = 250, window = 5, min_count = 5, workers = 40, iter = 8, sg = 1):
        self.w2v = Word2Vec(x, size = size, window = window, min_count = min_count, workers = workers, iter = iter, sg = sg)
        if save: self.w2v.save(path)

    def get_w2v(self, path = '', sen_len = 50):
        self.preprocess = self.Preprocess(x = self.train_X + self.unlabel + self.test_X, path = path, model = self.w2v, sen_len = sen_len)
        self.embedding = self.preprocess.make_embedding()

    def word2vec(self, x):
        return self.preprocess.sentence_word2idx(x)
    
    def get_validation_set(self, size = 0.05, seed = 64):
        np.random.seed(seed)
        shuf = np.arange(len(self.train_X))
        np.random.shuffle(shuf)
        size = int(len(self.train_X) * size)
        self.train_X, self.train_Y = self.train_X[shuf], self.train_Y[shuf]
        self.validate_X, self.validate_Y = self.train_X[:size], self.train_Y[:size]
        self.train_X, self.train_Y = self.train_X[size:], self.train_Y[size:]
        
    def create_data_set(self, x, y = None, batch_size = 128, shuffle = False, num_workers = 8):
        data_set = self.word_data_set(x, y)
        data_loader = DataLoader(data_set, batch_size, shuffle = shuffle, num_workers = num_workers)
        return data_loader
    
    def init(self, embedding_dim = 250, hidden_dim = 150, layers = 2, dropout = 0.5, train_embedding = False):
        self.Rnn = self.classifier(embedding = self.embedding, embedding_dim = embedding_dim, \
            hidden_dim = hidden_dim, layers = layers, dropout = dropout, train_embedding = train_embedding).cuda()

    class Preprocess:

        def __init__(self, x = None, model = None, path = '', sen_len = 50):
            if len(path): self.embedding = Word2Vec.load(path)
            else: self.embedding = model
            self.x = x
            self.dim = self.embedding.vector_size
            self.sen_len = sen_len
            self.idx2word = []
            self.word2idx = {}
            self.embedding_matrix = []

        def add_embedding(self, word):
            vector = torch.empty(1, self.dim)
            torch.nn.init.uniform_(vector)
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
        
        def make_embedding(self):
            for word in self.embedding.wv.vocab:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word.append(word)
                self.embedding_matrix.append(self.embedding.wv[word])
            self.embedding_matrix = torch.tensor(self.embedding_matrix)
            self.add_embedding('<PAD>')
            self.add_embedding('<UNK>')
            return self.embedding_matrix
        
        def pad_sentence(self, sentence):
            if len(sentence) > self.sen_len:
                sentence = sentence[:self.sen_len]
            else:
                pad_len = self.sen_len - len(sentence)
                for _ in range(pad_len):
                    sentence.append(self.word2idx['<PAD>'])
            return sentence

        def sentence_word2idx(self, x):
            lst = []
            for s in x:
                s_id = []
                for word in s:
                    if word in self.word2idx.keys(): s_id.append(self.word2idx[word])
                    else: s_id.append(self.word2idx['<UNK>'])
                s_id = self.pad_sentence(s_id)
                lst.append(s_id)
            return torch.LongTensor(lst)
        
    class word_data_set(Dataset):
        def __init__(self, X, Y = None):
            self.x, self.y = X, Y
            if self.y is not None:
                self.y = torch.LongTensor(self.y)
        
        def __len__(self):
            return len(self.x)
        
        def __getitem__(self, index):
            if self.y is not None: return self.x[index], self.y[index]
            else: return self.x[index]

    class classifier(nn.Module):

        def __init__(self, embedding, embedding_dim, hidden_dim, layers = 2, dropout = 0.5, train_embedding = False):
            super().__init__()
            self.embedding = nn.Embedding(embedding.size()[0], embedding.size()[1])
            self.embedding.weight = nn.Parameter(embedding)
            self.embedding.weight.requires_grad = train_embedding
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers, batch_first = True)
            self.post = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        
        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.lstm(x, None)
            x = x[:, -1, :]
            x = self.post(x)
            return x

torch.manual_seed(64)
torch.cuda.manual_seed_all(64)
mode = sys.argv[1]
model = RNN()
#model.get_train_data(label = False)
#model.train_w2v(model.train_X + model.test_X, path = './w2v_model', iter = 10, size = 200)
model.get_w2v('./w2v_model', sen_len = 40)
lst = ['./lstm_seed_2628', './lstm_seed_64', './lstm_seed_47', './lstm_seed_12712345', './lstm_seed_1126']
if mode == 'train':
    model.label_path, model.unlabel_path = sys.argv[2], sys.argv[3]
    for models in lst:
        model.get_train_data()
        model.train_X = model.word2vec(model.train_X)
        seed = int(models.split('_')[2])
        print(f'getting validation set with seed {seed}')
        model.get_validation_set(seed = seed, size = 0.01)
        model.init(train_embedding = False, embedding_dim = 200, hidden_dim = 100, dropout = 0.5, layers=2)
        model.train(epoch = 15, batch_size = 64, save_path = models)
elif mode == 'test':
    model.test_path, model.result_path = sys.argv[2], sys.argv[3]
    model.get_test_data()
    model.test(model_path = lst, result_path = model.result_path)
