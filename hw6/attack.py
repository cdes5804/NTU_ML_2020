import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd.gradcheck import zero_gradients
import matplotlib.pyplot as plt
import sys

device = torch.device("cuda")

class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        self.root = root
        self.label = torch.from_numpy(label).long()
        self.transforms = transforms
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        img = self.transforms(img)
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        return 200
        

class Attacker:
    def __init__(self, img_dir, label):
        self.model = models.densenet121(pretrained = True)
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        self.transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        self.dataset = Adverdataset(os.path.join(img_dir, 'images'), label, self.transform)
        
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)
    
    def deprocess(x):
        return np.clip((np.transpose(np.squeeze(x, axis = 0), (1, 2, 0)) * self.std + self.mean) * 255, 0, 255).astype('uint8')

    def fgsm_attack(self, image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image
    
    def attack(self, epsilon):
        adv_examples = []
        wrong, fail, success = 0, 0, 0
        raws = []
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            data_raw = data;
            data.requires_grad = True

            data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            data_raw = data_raw.squeeze().detach().cpu().numpy()
            raws.append(data_raw)

            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            if init_pred.item() != target.item():
                success += 1
                adv_examples.append(data_raw)
                continue
            
            loss = F.cross_entropy(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
       
            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
          
            if final_pred.item() == target.item():
                fail += 1
                adv_examples.append(data_raw)
            else:
                success += 1
                adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu().numpy()
                adv_examples.append(adv_ex)
            
        final_acc = (fail / (wrong + success + fail))
        
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, fail, len(self.loader), final_acc))
        return np.clip((np.transpose(adv_examples, (0, 2, 3, 1)) * 255.0).astype(np.uint8), 0, 255), np.clip((np.transpose(raws, (0, 2, 3, 1)) * 255.0).astype(np.uint8), 0, 255)

if __name__ == '__main__':
    img_dir = sys.argv[1]
    output_path = sys.argv[2]
    df = pd.read_csv(os.path.join(img_dir, 'labels.csv'))
    df = df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv(os.path.join(img_dir, 'categories.csv'))
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    attacker = Attacker(img_dir, df)
    epsilons = 0.01

    accuracies, examples = [], []

    ex, raw = attacker.attack(epsilons)

    print(np.mean(np.max(np.abs(np.array(ex.astype(np.int32) - raw.astype(np.int32))), axis=(-1, -2, -3))))
    
    for i, img in enumerate(ex):
        im = Image.fromarray(img)
        im.save(os.path.join(output_path, '{:03d}'.format(i) + '.png'))
    