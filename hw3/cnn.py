import os
import torch
import numpy as np
import cv2
import torch.nn as nn
import torchvision.transforms as transforms
import csv
from torch.utils.data import DataLoader, Dataset
import sys
import time

class CNN:

    def __init__(self, seed = 64):
        torch.manual_seed(seed)
        self.Cnn = self.classifier().cuda()
        self.Cnn.eval()
        self.get_transform()
        self.loss = nn.CrossEntropyLoss()

    def get_data(self, is_train = True, dir = ''):
        if dir[-1] != '/': dir += '/'
        if is_train:
            img_train = os.listdir(dir + 'training/')
            img_valid = os.listdir(dir + 'validation/')
            data = [dir + 'training/' + picture for picture in img_train] + \
                [dir + 'validation/' + picture for picture in img_valid]
        else:
            img_test = sorted(os.listdir(dir + 'testing/'))
            data = [dir + 'testing/' + picture for picture in img_test]
        X = np.zeros((len(data), 192, 192, 3), dtype = np.uint8)
        if is_train:
            Y = np.zeros((len(data)), dtype=np.uint8)
        counter = 0
        for picture in data:
            img = cv2.imread(picture)
            X[counter, :, :] = cv2.resize(img, (192, 192))
            if is_train:
                Y[counter] = int(picture.split('/')[-1].split('_')[0])
            counter += 1
        if is_train:
            self.train_X, self.train_Y = X, Y
        else:
            self.test_X = X
    
    def get_validation_set(self, size = 3000, seed = 64):
        np.random.seed(seed)
        shuf = np.arange(len(self.train_X))
        np.random.shuffle(shuf)
        self.train_X, self.train_Y = self.train_X[shuf], self.train_Y[shuf]
        self.validate_X, self.validate_Y = self.train_X[:size], self.train_Y[:size]
        self.train_X, self.train_Y = self.train_X[size:], self.train_Y[size:]

    def get_transform(self):
        self.train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness = (0.5, 1.5), contrast = (0.5, 1.5), saturation = (0.5, 1.5)),
        transforms.RandomPerspective(),
        transforms.RandomAffine(15),
        transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
        transforms.ToPILImage(),                                    
        transforms.ToTensor(),
        ])

    class image_data_set(Dataset):

        def __init__(self, X, Y = None, transform = None):
            self.x = X
            self.y = Y
            if self.y is not None:
                self.y = torch.LongTensor(self.y)
            self.transform = transform
        
        def __len__(self):
            return len(self.x)

        def __getitem__(self, index):
            X = self.x[index]
            if self.transform is not None:
                X = self.transform(X)
            if self.y is not None:
                Y = self.y[index]
                return X, Y
            else:
                return X

    def create_data_set(self, is_train = True, has_validation = True, batch_size = 64):
        if is_train:
            self.train_set = self.image_data_set(self.train_X, self.train_Y, self.train_transform)
            self.train_loader = DataLoader(self.train_set, batch_size, shuffle = True, num_workers = 4)
            if has_validation:
                self.valid_set = self.image_data_set(self.validate_X, self.validate_Y, self.test_transform)
                self.valid_loader = DataLoader(self.valid_set, batch_size, shuffle = False, num_workers = 4)
        else:
            self.test_set = self.image_data_set(self.test_X, transform = self.test_transform)
            self.test_loader = DataLoader(self.test_set, batch_size, shuffle = False, num_workers = 4)

    class classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding = 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 3, padding = 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(128),

                nn.MaxPool2d(2, 2, 0),
                nn.Dropout(p = 0.1),
                
                nn.Conv2d(128, 256, 3, padding = 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, 3, padding = 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(256),

                nn.MaxPool2d(2, 2, 0),
                nn.Dropout(p = 0.2),

                nn.Conv2d(256, 256, 3, padding = 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, 3, padding = 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(256),

                nn.MaxPool2d(2, 2, 0),
                nn.Dropout(p = 0.2),

                nn.Conv2d(256, 512, 3, padding = 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 512, 3, padding = 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(512),

                nn.MaxPool2d(2, 2, 0),
                nn.Dropout(p = 0.2),
                
                nn.Conv2d(512, 512, 3, padding = 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 512, 3, padding = 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(512),

                nn.MaxPool2d(2, 2, 0),
                nn.Dropout(p = 0.2),
            )
            self.fc = nn.Sequential(
            nn.Linear(512*6*6, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 11),
            )
        def forward(self, x):
            prel = self.cnn(x)
            prel = prel.view(prel.size()[0], -1)
            return self.fc(prel)
        
    def train(self, validation_size = 3000, epoch = 100, seed = 64, batch_size = 64):
        self.optimizer = torch.optim.Adam(self.Cnn.parameters(), lr = 0.001)
        self.epoch = epoch
        self.get_data(is_train = True, dir = sys.argv[2])
        if validation_size:
            self.get_validation_set(validation_size, seed)
        self.create_data_set(is_train = True, has_validation = validation_size != 0, batch_size = batch_size)
        for step in range(self.epoch):
            current_time = time.time()
            train_acc, train_loss, validate_acc, validate_loss = 0, 0, 0, 0
            self.Cnn.train()
            for Data, label in self.train_loader:
                self.optimizer.zero_grad()
                pred = self.Cnn(Data.cuda())
                Loss = self.loss(pred, label.cuda())
                Loss.backward()
                self.optimizer.step()
                train_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis = 1) == label.numpy())
                train_loss += Loss.item()
            if validation_size:
                self.Cnn.eval()
                with torch.no_grad():
                    for Data, label in self.valid_loader:
                        pred = self.Cnn(Data.cuda())
                        Loss = self.loss(pred, label.cuda())
                        validate_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis = 1) == label.numpy())
                        validate_loss += Loss.item()
            print(time.time() - current_time)
            print(f'epoch: {step + 1}')
            print(f'train_acc: {train_acc / self.train_set.__len__()} train_loss: {train_loss / self.train_set.__len__()}')
            if validation_size:
                print(f'validate_acc: {validate_acc / self.valid_set.__len__()}, validate_loss: {validate_loss / self.valid_set.__len__()}')

    def test(self, result_file, batch_size = 40):
        self.Cnn.eval()
        self.get_data(is_train = False, dir = sys.argv[2])
        self.create_data_set(is_train = False, batch_size = batch_size)
        output = [['Id', 'Category']]
        counter = 0
        with torch.no_grad():
            for Data in self.test_loader:
                pred = self.Cnn(Data.cuda())
                label = np.argmax(pred.cpu().data.numpy(), axis = 1)
                for y in label:
                    output.append([counter, y])
                    counter += 1
        csv.writer(open(result_file, 'w')).writerows(output)

    def save(self, path):
        torch.save(self.Cnn.state_dict(), path)
    
    def load(self, path):
        self.Cnn.load_state_dict(torch.load(path))

model = CNN(seed = 127)
if sys.argv[1] == 'train':
    model.train(validation_size = 0, epoch = 220, batch_size = 24)
    model.save('./model')
elif sys.argv[1] == 'test':
    model.load('./model')
    model.test(sys.argv[3], batch_size = 20)
else:
    print('No argument given(train/test)')