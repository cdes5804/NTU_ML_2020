import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
import sys

def preprocess(image_list):
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list


class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images

trainX = np.load(sys.argv[2])
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)

def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.PReLU(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.PReLU(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.PReLU(256),
            nn.MaxPool2d(2)
        )
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.PReLU(128),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.PReLU(64),
            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x  = self.decoder(x1)
        return x1, x

same_seeds(64)

model = AE().cuda()
if sys.argv[1] == 'test':
    model.load_state_dict(torch.load(sys.argv[3]))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

model.train()
n_epoch = 100

img_dataloader = DataLoader(img_dataset, batch_size=32, shuffle=True)

if sys.argv[1] == 'train':
    for epoch in range(n_epoch):
        for data in img_dataloader:
            img = data
            img = img.cuda()

            output1, output = model(img)
            loss = criterion(output, img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
                
        print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, loss.data))

    torch.save(model.state_dict(), sys.argv[3])

def cal_acc(gt, pred):
    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    return max(acc, 1-acc)

def plot_scatter(feat, label, savefig=None):
    X = feat[:, 0]
    Y = feat[:, 1]
    plt.scatter(X, Y, c = label)
    plt.legend(loc='best')
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()

def inference(X, model, batch_size=256):
    X = preprocess(X)
    dataset = Image_Dataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)
    print('Latents Shape:', latents.shape)
    return latents

def predict(latents):
    transformer = KernelPCA(n_components=100, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)

    X_embedded = TSNE(n_components=2, n_jobs = -1).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    pred = MiniBatchKMeans(n_clusters=2, random_state=0, n_init=64, max_iter = 1000).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred

def invert(pred):
    return np.abs(1-pred)

def save_prediction(pred, out_csv='prediction.csv'):
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'prediction saved to {out_csv}.')


if sys.argv[1] == 'test':
    model.eval()

    latents = inference(X=trainX, model = model)
    pred = predict(latents)
    save_prediction(invert(pred), sys.argv[4])


