from typing import Dict, Optional, Tuple
from sympy import Ci
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm_guided import DDPM

import h5py
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import wandb

class MagnetismData(Dataset):
    def __init__(self, db, transform=None):
        self.db = db
        self.transform = transform

    def __len__(self):
        return self.db['field'].shape[0]

    def __getitem__(self, idx):
        field = self.db['field'][idx].transpose(1,2,0)
        if self.transform:
            field = self.transform(field)
        return field
    
def curl_2d(field) -> np.ndarray:
    Fx_y = np.gradient(field[0], axis=1)
    Fy_x = np.gradient(field[1], axis=0)
    return Fy_x - Fx_y

def div_2d(field) -> np.ndarray:
    Fx_x = np.gradient(field[0], axis=0)
    Fy_y = np.gradient(field[1], axis=1)
    div = np.stack([Fx_x, Fy_y], axis=0)

    return div.sum(axis=0)

def train_magnet(
    epochs: int = 101, betas: tuple = (1e-4, 0.02), n_T: int = 1000, features: int = 64, lr: float = 2e-4,
    batch_size: int = 128,
    device: str = "cuda:1"#, load_pth: Optional[str] = None
) -> None:
    
    dbpath = '/home/s214435/data/magfield_symm_64_30000.h5'

    cfg = {"epochs": epochs, "betas": betas, "n_T": n_T, 
                           "features":features, "lr":lr, "batch_size":batch_size, "db":dbpath}

    wandb.init(
        # set the wandb project where this run will be logged
        entity="dl4mag",
        project="mag-diffusion",

        # track hyperparameters and run metadata
        config=cfg
    )
    

    db = h5py.File(dbpath)
    dbstd = np.std(db['field'])
    channels = db["field"].shape[1]

    ddpm = DDPM(eps_model=NaiveUnet(channels, channels, n_feat=cfg["features"]), betas=cfg["betas"], n_T=cfg["n_T"])
    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0, )*channels, (dbstd, )*channels)]
    )

    dataset = MagnetismData(
    db,
    transform=tf
    )

    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=20)
    optim = torch.optim.Adam(ddpm.parameters(), lr=cfg["lr"])

    for i in range(cfg["epochs"]):
        print(f"Epoch {i} : ")
        ddpm.train()


        #pbar = tqdm(dataloader)
        loss_ema = None
        for x in dataloader:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            #pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        if (i % 5 == 0):
            ddpm.eval()
            with torch.no_grad():
                samples = 4
                xh = ddpm.sample(samples, (channels, 64, 64), device)
                fig, axes = plt.subplots(nrows=samples, ncols=channels, sharex=True,
                                        sharey=True, figsize=(8,8))
                norm = colors.Normalize(vmin=-1, vmax=1)
                
                tot_curl = 0
                tot_div = 0

                for j, sam in enumerate(xh):
                    sam_field = sam.cpu()
                    tot_curl = tot_curl + abs(curl_2d(sam_field.numpy()*dbstd)).mean()
                    tot_div = tot_div + abs(div_2d(sam_field.numpy())*dbstd).mean()

                    for k, comp in enumerate(sam_field):
                        #img = comp.permute(1,2,0)
                        ax = axes.flat[j*channels+k]
                        im = ax.imshow(comp.numpy(), cmap='bwr', norm=norm, origin="lower")

                cbar_ax = fig.add_axes([0.9, 0.345, 0.015, 0.3])
                fig.colorbar(im, cax=cbar_ax)
                #fig.savefig(f"./contents/ddpm_sample_{i}.png")
                print(f"Loss: {loss_ema} | Div: {tot_div/samples} | Curl: {tot_curl/samples}")

                wandb.log({"loss": loss_ema, "avg_curl":tot_curl/samples, "avg_div": tot_div/samples, "sample":wandb.Image(fig)})

                plt.close()

                # save model
                torch.save(ddpm.state_dict(), f"./ddpm_magnet.pth")
        else:
            print(f"Loss: {loss_ema}")
            wandb.log({"loss": loss_ema})

    wandb.save(f"./ddpm_magnet.pth")
    wandb.finish()


if __name__ == "__main__":
    train_magnet(epochs= 101, betas= (1e-4, 0.02), n_T= 1000, 
                           features=128, lr=1e-5, batch_size=128 )
