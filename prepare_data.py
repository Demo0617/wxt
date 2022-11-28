from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os


class MyDataset(Dataset):
    def __init__(self, data_dir, preprocess):
        captions = []
        names = []
        imgs = []
        with open('{}/captions.txt'.format(data_dir)) as f:
            for l in f.readlines()[1:]:
                name = l.split('.')[0]
                captions.append(l.split(',')[-1].rstrip('\n'))
                names.append(name)
                imgs.append(preprocess(Image.open(os.path.join(data_dir, 'Images', name+'.jpg')).convert("RGB")))
        self.cap = captions
        self.names = names
        self.imgs = imgs

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.cap[idx], self.imgs[idx], self.names[idx]