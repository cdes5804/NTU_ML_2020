import numpy as np
import os
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from torchvision.models import densenet121
from PIL import Image
import sys
import pandas as pd

def DeprocessImg(x, mean, std):
    return np.clip((np.transpose(np.squeeze(x, axis = 0), (1, 2, 0)) * std + mean) * 255, 0, 255).astype('uint8')

def get_data(input_path):
    df = pd.read_csv(os.path.join(input_path, 'labels.csv'))
    df = df.loc[:, 'TrueLabel'].to_numpy()
    return df

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    labels = get_data(input_dir)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    model = densenet121(pretrained = True).cuda()
    model.eval()

    trans = transform.Compose([transform.ToTensor(), transform.Normalize(mean = mean, std = std)])
    fail = 0
    Loss = nn.CrossEntropyLoss().cuda()
    for idx in range(200):
        input = Image.open(os.path.join(input_dir, 'images',  "%03d.png" % (idx)))
        input_tensor = trans(input).unsqueeze(0).cuda()
        input_tensor.requires_grad = True
        input_label = np.argmax(model(input_tensor).cpu().detach().numpy())

        for eps in range(256):
            zero_gradients(input_tensor)

            pred = model(input_tensor)
            loss = Loss(pred, torch.tensor([labels[idx]]).cuda())
            loss.backward()

            output_tensor = input_tensor + (eps / 256) * input_tensor.grad.sign_()
            output_label = np.argmax(model(trans(DeprocessImg(output_tensor.detach().cpu().numpy(), mean, std)).unsqueeze(0).cuda()).cpu().detach().numpy())

            if input_label != output_label:
                im = Image.fromarray(DeprocessImg(output_tensor.detach().cpu().numpy(), mean, std))
                im.save("%s/%03d.png" % (output_dir, idx))
                #imsave("%s/%03d.png" % (output_dir, idx), DeprocessImg(output_tensor.detach().cpu().numpy(), mean, std))
                print("%s/%03d.png success with %d" % (input_dir, idx, eps))
                break
        else:
            fail += 1
            print("%s/%03d.png failed" % (input_dir, idx))
            input.save("%s/%03d.png" % (output_dir, idx))
            #imsave("%s/%03d.png" % (output_dir, idx), input)
    print(fail / 200)

