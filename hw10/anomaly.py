import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, pairwise_distances, roc_auc_score
from scipy.cluster.vq import vq, kmeans
import sys
import os
import csv
import torchvision
import torchvision.transforms as transforms
import random
seed = 862765349
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_result(pred, path = 'predict.csv'):
    output = [['id', 'anomaly']]
    for i, data in enumerate(pred):
        output.append([i + 1, data])
    csv.writer(open(path, 'w')).writerows(output)

class BaseLineAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 12, 4, stride = 2, padding = 1),
                nn.ReLU(),
                nn.Conv2d(12, 24, 4, stride = 2, padding = 1),
                nn.ReLU(),
                nn.Conv2d(24, 48, 4, stride = 2, padding = 1),
                nn.ReLU(),
                )
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(48, 24, 4, stride = 2, padding = 1),
                nn.ReLU(),
                nn.ConvTranspose2d(24, 12, 4, stride = 2, padding = 1),
                nn.ReLU(),
                nn.ConvTranspose2d(12, 3, 4, stride = 2, padding = 1),
                nn.Tanh(),
                )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
                nn.PReLU(32),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 32, kernel_size = 5, stride = 2, padding = 2),
                nn.PReLU(32),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size = 5, stride = 2, padding = 2),
                nn.PReLU(64),
                nn.BatchNorm2d(64),
                )
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
                nn.BatchNorm2d(3),
                nn.Tanh(),
               )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def test(test_data, model_path, predict_path):
    model = BaseLineAE().cuda() if 'baseline' in model_path else AE().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_set = TensorDataset(torch.tensor(test_data, dtype = torch.float))
    test_loader = DataLoader(test_set, batch_size = 32, shuffle = False, num_workers = 12)
    with torch.no_grad():
        tr, res = [], []
        for i, x in enumerate(test_loader):
            img = x[0].transpose(3, 1).cuda()
            output = model(img)
            output = output.transpose(3, 1)
            res.append(output.cpu().detach().numpy())

        res = np.concatenate(res, axis = 0)
        pred = np.sqrt(np.sum(np.square(res - test_data).reshape(len(test_data), -1), axis = 1))
    get_result(pred, path = predict_path)

def train(train_data, model_path, epoch = 101):
    model = BaseLineAE().cuda() if 'baseline' in model_path else AE().cuda()
    train_set = TensorDataset(torch.tensor(train_data, dtype = torch.float))
    train_loader = DataLoader(train_set, batch_size = 32, shuffle = True, num_workers = 12)
    Loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    for step in range(epoch):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        model.train()
        total_loss = 0
        for x in train_loader:
            optimizer.zero_grad()
            img = x[0].transpose(3, 1).cuda()
            output = model(img)
            loss = Loss(output, img)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'epoch [{step + 1}/{epoch}], loss:{total_loss / len(train_loader)}')
    torch.save(model.state_dict(), model_path)

if sys.argv[1] == 'train':
    train_data = np.load(sys.argv[2])
    train(train_data, sys.argv[3],  epoch = 101)
if sys.argv[1] == 'test':
    test_data = np.load(sys.argv[2])
    test(test_data, sys.argv[3], sys.argv[4])


