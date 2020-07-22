import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace
import shap
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2

class Explainable:
    
    def __init__(self):
        self.Cnn = self.classifier().cuda()
        self.Cnn.eval()
        self.loss = nn.CrossEntropyLoss()
        self.dir = sys.argv[1]
        self.target = sys.argv[2]
        self.train_paths, self.train_labels = self.get_paths_labels(os.path.join(self.dir, 'training'))
        self.train_set = self.image_data_set(self.train_paths, self.train_labels, mode='eval')

    def normalize(self, image):
        return (image - image.min()) / (image.max() - image.min())

    def get_paths_labels(self, path):
        imgnames = os.listdir(path)
        imgnames.sort()
        imgpaths = []
        labels = []
        for name in imgnames:
            imgpaths.append(os.path.join(path, name))
            labels.append(int(name.split('_')[0]))
        return imgpaths, labels
    
    def compute_saliency_maps(self, x, y, model):
        model.eval()
        x = x.cuda()
        x.requires_grad_()

        y_pred = model(x)
        loss = self.loss(y_pred, y.cuda())
        loss.backward()

        saliencies = x.grad.abs().detach().cpu()
        saliencies = torch.stack([self.normalize(item) for item in saliencies])
        return saliencies

    def get_saliency(self, img_indices = []):
        images, labels = self.train_set.getbatch(img_indices)
        saliencies = self.compute_saliency_maps(images, labels, self.Cnn)

        _, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
        for row, target in enumerate([images, saliencies]):
            for column, img in enumerate(target):
                axs[row][column].imshow(img.permute(1, 2, 0).numpy())
        path = os.path.join(self.target, 'saliency.png')
        plt.savefig(path)
        plt.close()
    
    def filter_explaination(self, x, model, cnnid, filterid, iteration=100, lr=1):
        model.eval()
        layer_activations = None

        def hook(model, input, output):
            nonlocal layer_activations
            layer_activations = output

        hook_handle = model.cnn[cnnid].register_forward_hook(hook)
        model(x.cuda())
        filter_activations = layer_activations[:, filterid, :, :].detach().cpu()
        x = torch.rand(1, 3, 192, 192).cuda()
        x.requires_grad_()
        optimizer = Adam([x], lr=lr)
        for _ in range(iteration):
            optimizer.zero_grad()
            model(x)
            objective = -layer_activations[:, filterid, :, :].abs().sum()
            objective.backward()
            optimizer.step()
        filter_visualization = x.detach().cpu().squeeze()

        hook_handle.remove()

        return filter_activations, filter_visualization
    
    def get_filter_visual(self, img_indices = [], cnnid = 0, filterid = 0, iteration = 1000, lr = 0.01):
        images, _ = self.train_set.getbatch(img_indices)
        filter_activations, filter_visualization = self.filter_explaination(images, self.Cnn, cnnid, filterid, iteration, lr)

        plt.figure(figsize=(15, 8))
        plt.imshow(self.normalize(filter_visualization.permute(1, 2, 0)))
        path = os.path.join(self.target, f'{cnnid}_visual.png')
        plt.savefig(path)
        _, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
        for i, img in enumerate(images):
            axs[0][i].imshow(img.permute(1, 2, 0))
        for i, img in enumerate(filter_activations):
            axs[1][i].imshow(self.normalize(img))
        path = os.path.join(self.target, f'{cnnid}_active.png')
        plt.savefig(path)
        plt.close()
    
    def get_lime(self, img_indices = []):

        def predict(input):
            self.Cnn.eval()
            input = torch.FloatTensor(input).permute(0, 3, 1, 2)

            output = self.Cnn(input.cuda())
            return output.detach().cpu().numpy()

        def segmentation(input):
            return slic(input, n_segments=100, compactness=1, sigma=1)

        images, labels = self.train_set.getbatch(img_indices)
        _, axs = plt.subplots(1, 5, figsize=(15, 8))
        np.random.seed(64)
        for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
            x = image.astype(np.double)
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation, top_labels = 11)
            lime_img, _ = explanation.get_image_and_mask(
                                        label=label.item(),
                                        positive_only=False,
                                        hide_rest=False,
                                        num_features=11,
                                        min_weight=0.05
                                    )
            axs[idx].imshow(lime_img)

        path = os.path.join(self.target, 'lime.png')
        plt.savefig(path)
        plt.close()

    def get_shap(self, indices = []):
        np.random.seed(64)
        tr = np.random.randint(1000, size=20)
        image, _ = self.train_set.getbatch(tr)
        e = shap.DeepExplainer(self.Cnn, image.cuda())
        test_images = self.train_set.getbatch(indices)[0].cuda()
        shap_values = e.shap_values(test_images)
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)
        shap.image_plot(shap_numpy, -test_numpy, show=False)
        path = os.path.join(self.target, 'shap.png')
        plt.savefig(path)
        plt.close()
    
    def load(self, path):
        self.Cnn.load_state_dict(torch.load(path))

    class image_data_set(Dataset):

        def __init__(self, paths, labels, mode):
            self.paths = paths
            self.labels = labels
            trainTransform = transforms.Compose([
                transforms.Resize(size=(192,192)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness = (0.5, 1.5), contrast = (0.5, 1.5), saturation = (0.5, 1.5)),
                transforms.RandomPerspective(),
                transforms.RandomAffine(15),
                transforms.ToTensor(),
            ])
            evalTransform = transforms.Compose([
                transforms.Resize(size=(192,192)),                               
                transforms.ToTensor(),
            ])
            self.transform = trainTransform if mode == 'train' else evalTransform
        
        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            X = Image.open(self.paths[index])
            X = self.transform(X)
            Y = self.labels[index]
            return X, Y
        
        def getbatch(self, indices):
            images = []
            labels = []
            for index in indices:
                image, label = self.__getitem__(index)
                images.append(image)
                labels.append(label)
            return torch.stack(images), torch.tensor(labels)
    
    class classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding = 1), #1
                nn.ReLU(), #2
                nn.BatchNorm2d(64), #3

                nn.MaxPool2d(2, 2, 0), #4
                nn.Dropout(p = 0.1), #5

                nn.Conv2d(64, 128, 3, padding = 1), #6
                nn.ReLU(), #7
                nn.BatchNorm2d(128), #8

                nn.MaxPool2d(2, 2, 0), #9
                nn.Dropout(p = 0.2), #10
                
                nn.Conv2d(128, 256, 3, padding = 1), #11
                nn.ReLU(), #12
                nn.BatchNorm2d(256), #13

                nn.MaxPool2d(2, 2, 0), #14
                nn.Dropout(p = 0.2), #15

                nn.Conv2d(256, 256, 3, padding = 1), #16
                nn.ReLU(), #17
                nn.BatchNorm2d(256), #18

                nn.MaxPool2d(2, 2, 0), #19
                nn.Dropout(p = 0.2), #20

                nn.Conv2d(256, 512, 3, padding = 1), #21
                nn.ReLU(), #22
                nn.BatchNorm2d(512), #23
                nn.Conv2d(512, 512, 3, padding = 1), #24
                nn.ReLU(), #25
                nn.BatchNorm2d(512), #26

                nn.MaxPool2d(2, 2, 0), #27
                nn.Dropout(p = 0.2), #28

            )
            self.fc = nn.Sequential(
            nn.Linear(512*6*6, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(),
            nn.Linear(1024, 11),
            )
        def forward(self, x):
            prel = self.cnn(x)
            prel = prel.view(prel.size()[0], -1)
            return self.fc(prel)
        
    
model = Explainable()
model.load('./model')
indices = [5465, 8365, 7229, 2131, 9864]
model.get_saliency(indices)
model.get_filter_visual(indices, cnnid = 1, filterid = 0)
model.get_filter_visual(indices, cnnid = 6, filterid = 0)
model.get_filter_visual(indices, cnnid = 11, filterid = 150)
model.get_filter_visual(indices, cnnid = 16, filterid = 100)
model.get_lime(indices)
model.get_shap(indices)