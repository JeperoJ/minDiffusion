from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import matplotlib.pyplot as plt

import h5py
import os
from torchvision.io import read_image
from torch.utils.data import Dataset

class MagnetismData(Dataset):
    def __init__(self, db, transform=None):
        self.db = db
        self.transform = transform

    def __len__(self):
        return len(self.db['field'])

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        field = self.db['field'][idx][0]
        if self.transform:
            field = self.transform(field)
        return field

tf1 = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )

tf2 = transforms.Compose(
        [transforms.ToTensor()]
    )

dataset1 = MNIST(
        "./data",
        train=True,
        download=False,
        transform=tf1,
    )

dataset2 = MagnetismData(
    h5py.File('/home/s214435/data/magfield_32.h5'),
    transform=tf2
)

dataloader = DataLoader(dataset2, batch_size=128, shuffle=True, num_workers=20)

pbar = tqdm(dataloader)
for x in pbar:
    pbar.set_description(f"loss: {0.5000:.4f}")
    print("Yeet")
    print(x)