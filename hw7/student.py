import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import csv
import sys

torch.manual_seed(127)
np.random.seed(127)

class StudentNet(nn.Module):
    def __init__(self, base = 16, width_mult=1):

            super().__init__()
            multiplier = [2, 4, 8, 8, 16, 16, 16, 16]
            bandwidth = [ base * m for m in multiplier]

            for i in range(3, len(multiplier)):
                bandwidth[i] = int(bandwidth[i] * width_mult)

            self.cnn = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                    nn.BatchNorm2d(bandwidth[0]),
                    nn.ReLU6(),
                    nn.MaxPool2d(2, 2, 0),
                ),
                nn.Sequential(
                    nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),
                    nn.BatchNorm2d(bandwidth[0]),
                    nn.ReLU6(),
                    nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                    nn.MaxPool2d(2, 2, 0),
                ),

                nn.Sequential(
                    nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),
                    nn.BatchNorm2d(bandwidth[1]),
                    nn.ReLU6(),
                    nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                    nn.MaxPool2d(2, 2, 0),
                ),

                nn.Sequential(
                    nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),
                    nn.BatchNorm2d(bandwidth[2]),
                    nn.ReLU6(),
                    nn.Conv2d(bandwidth[2], bandwidth[3], 1),
                    nn.MaxPool2d(2, 2, 0),
                ),
                nn.Sequential(
                    nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),
                    nn.BatchNorm2d(bandwidth[3]),
                    nn.ReLU6(),
                    nn.Conv2d(bandwidth[3], bandwidth[4], 1),
                ),

                nn.Sequential(
                    nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),
                    nn.BatchNorm2d(bandwidth[4]),
                    nn.ReLU6(),
                    nn.Conv2d(bandwidth[4], bandwidth[5], 1),
                ),

                nn.Sequential(
                    nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),
                    nn.BatchNorm2d(bandwidth[5]),
                    nn.ReLU6(),
                    nn.Conv2d(bandwidth[5], bandwidth[6], 1),
                ),

                nn.Sequential(
                    nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),
                    nn.BatchNorm2d(bandwidth[6]),
                    nn.ReLU6(),
                    nn.Conv2d(bandwidth[6], bandwidth[7], 1),
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc = nn.Sequential(
                nn.Linear(bandwidth[7], bandwidth[7] // 4),
                nn.Linear(bandwidth[7] // 4, 11),
            )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None, test = False):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in sorted(os.listdir(folderName)):
            class_idx = int(img_path.split('_')[0]) if not test else -1
            self.data.append(os.path.join(folderName, img_path))
            self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self.data[idx])
        image_fp = image.fp
        image.load()
        image_fp.close()
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]


trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def encode8(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))
    print(f"8-bit cost: {os.stat(fname).st_size} bytes.")


def decode8(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        custom_dict[name] = param

    return custom_dict


def get_dataloader(dir, mode='training', batch_size=24, train = True, label = True):

    assert mode in ['training', 'testing', 'validation']

    dataset = MyDataset(
        os.path.join(dir, mode),
        transform=trainTransform if train else testTransform,
        test = not label)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle = train)

    return dataloader

student_net = StudentNet().cuda()
dir = sys.argv[2]
if sys.argv[1] == 'train':
    train_dataloader = get_dataloader(dir = dir, mode = 'training', batch_size=32)
    valid_dataloader = get_dataloader(dir = dir, mode = 'validation', batch_size=32, train = False)
    teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
    teacher_net.load_state_dict(torch.load(f'teacher_resnet18.bin'))
    optimizer = optim.SGD(student_net.parameters(), lr=0.01, momentum = 0.9, nesterov = True)
if sys.argv[1] == 'test':
    test_dataloader = get_dataloader(dir = dir, mode = 'testing', batch_size = 32, train = False, label = False)



def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss

def run_epoch(dataloader, update=True, alpha=0.5):

    total_num, total_hit, total_loss = 0, 0, 0
    for batch_data in dataloader:
        optimizer.zero_grad()
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)

        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num

def get_result(dataloader, path):
    counter = 0
    output = [['Id', 'label']]
    with torch.no_grad():
        for batch_data in dataloader:
            inputs, _ = batch_data
            inputs = inputs.cuda()
            logits = student_net(inputs)
            label = np.argmax(logits.cpu().data.numpy(), axis = 1)
            for y in label:
                output.append([counter, y])
                counter += 1
        csv.writer(open(path, 'w')).writerows(output)


def model_compress(path, fname):
    param = torch.load(path)
    encode8(param, fname)


if sys.argv[1] == 'train':
    teacher_net.eval()
    now_best_acc = 0
    for epoch in range(400):
        student_net.train()
        train_loss, train_acc = run_epoch(train_dataloader, update=True)
        student_net.eval()
        valid_loss, valid_acc = run_epoch(valid_dataloader, update=False)

        if valid_acc > now_best_acc:
            now_best_acc = valid_acc
            torch.save(student_net.state_dict(), 'model.bin')
        print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f}, valid loss: {:6.4f}, acc {:6.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc))

    model_compress('model.bin', 'model.pkl')
if sys.argv[1] == 'test':
    para = decode8('model.pkl')
    student_net.load_state_dict(para)
    student_net.eval()
    get_result(test_dataloader, sys.argv[3])
