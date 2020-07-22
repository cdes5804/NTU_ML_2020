import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
import sys
import random
import os

seed = 64
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

source_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    ])
target_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    ])

train_dir = os.path.join(sys.argv[1], 'train_data')
test_dir = os.path.join(sys.argv[1], 'test_data')
#source_dataset = ImageFolder(train_dir, transform = source_transform)
target_dataset = ImageFolder(test_dir, transform = target_transform)

#source_dataloader = DataLoader(source_dataset, batch_size = 32, shuffle = True)
#target_dataloader = DataLoader(target_dataset, batch_size = 32, shuffle = True)
test_dataloader = DataLoader(target_dataset, batch_size = 128, shuffle = False)

class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.PReLU(64),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride = 2, padding = 1),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.PReLU(128),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride = 2, padding = 1),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.PReLU(256),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride = 2, padding = 1),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.PReLU(512),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, stride = 2, padding = 1),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.PReLU(512),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, stride = 2, padding = 1),
            )
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LabelPredictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
                nn.Linear(512, 512),
                nn.PReLU(),
                nn.Linear(512, 512),
                nn.PReLU(),
                nn.Linear(512, 10),
                )
    def forward(self, x):
        y = self.layer(x)
        return y

class DomainClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
                nn.Linear(512,512),
                nn.BatchNorm1d(512),
                nn.PReLU(),

                nn.Linear(512,512),
                nn.BatchNorm1d(512),
                nn.PReLU(),

                nn.Linear(512,512),
                nn.BatchNorm1d(512),
                nn.PReLU(),

                nn.Linear(512,512),
                nn.BatchNorm1d(512),
                nn.PReLU(),

                nn.Linear(512, 1)
                )
    def forward(self, x):
        y = self.layer(x)
        return y

feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
feature_extractor.load_state_dict(torch.load('feature_extractor.pth'))
label_predictor.load_state_dict(torch.load('label_predictor.pth'))
#domain_classifier = DomainClassifier().cuda()

#class_loss = nn.CrossEntropyLoss()
#domain_loss = nn.BCEWithLogitsLoss()

#optim_F = optim.Adam(feature_extractor.parameters())
#optim_C = optim.Adam(label_predictor.parameters())
#optim_D = optim.Adam(domain_classifier.parameters())

def train_epoch(source_dataloader, target_dataloader, p):
    D_loss, F_loss = 0, 0
    hit, num = 0, 0
    gamma = 10
    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        lamb = 2 / (1 + np.exp(-gamma*p)) - 1
        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()

        mixed_data = torch.cat([source_data, target_data], dim = 0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        domain_label[:source_data.shape[0]] = 1

        feature = feature_extractor(mixed_data)
        domain_logit = domain_classifier(feature.detach())
        loss = domain_loss(domain_logit, domain_label)
        D_loss += loss.item()
        loss.backward()
        optim_D.step()

        class_logit = label_predictor(feature[:source_data.shape[0]])
        domain_logit = domain_classifier(feature)
        loss = class_loss(class_logit, source_label) - lamb * domain_loss(domain_logit, domain_label)
        F_loss += loss.item()
        loss.backward()
        optim_F.step()
        optim_C.step()

        optim_D.zero_grad()
        optim_C.zero_grad()
        optim_F.zero_grad()

        hit += torch.sum(torch.argmax(class_logit, dim = 1) == source_label).item()
        num += source_data.shape[0]
        print(i, end = '\r')
    return D_loss / (i + 1), F_loss / (i + 1), hit / num

def train(epoch = 2000):
    for e in range(epoch):
        D_loss, F_loss, acc = train_epoch(source_dataloader, target_dataloader, p = e / epoch)
        print(f'epoch: {e + 1}, D loss: {D_loss}, F_loss: {F_loss}, acc: {acc}')

    torch.save(feature_extractor.state_dict(), 'feature_extractor.pth')
    torch.save(label_predictor.state_dict(), 'label_predictor.pth')

def test(file_path):
    result = []
    label_predictor.eval()
    feature_extractor.eval()
    for i, (test_data, _) in enumerate(test_dataloader):
        test_data = test_data.cuda()
        class_logits = label_predictor(feature_extractor(test_data))
        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        result.append(x)

    result = np.concatenate(result)
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv(file_path,index=False)
path = sys.argv[2]
test(path)

