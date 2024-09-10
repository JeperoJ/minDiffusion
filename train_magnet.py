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
from mindiffusion.ddpm import DDPM

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
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        field = self.db['field'][idx].transpose(1,2,0)
        #print(field.shape)
        if self.transform:
            field = self.transform(field)
        return field
    
def curl(field):
    # The magnetic field coming from magtense has
    # y-direction in the first dimension
    # x-compenent in the second dimension
    # Similar to 'xy' indexing in np.meshgrid
    Fx_y = np.gradient(field[0], axis=0)
    Fy_x = np.gradient(field[1], axis=1)

    if field.shape[-1]== 4:
        Fx_z = np.gradient(field[0], axis=2)
        Fy_z = np.gradient(field[1], axis=2)
        Fz_x = np.gradient(field[2], axis=1)
        Fz_y = np.gradient(field[2], axis=0)
        # Taking gradients of center layer only
        curl_vec = np.stack([
            Fz_y - Fy_z,
            Fx_z - Fz_x,
            Fy_x - Fx_y], axis=0)[:,:,:,1]
    else:
        curl_vec = Fy_x - Fx_y
    
    return curl_vec


def div(field):
    # The magnetic field coming from magtense has
    # y-direction in the first dimension
    # x-compenent in the second dimension
    # Similar to 'xy' indexing in np.meshgrid
    Fx_x = np.gradient(field[0], axis=1)
    Fy_y = np.gradient(field[1], axis=0)

    if field.shape[-1] == 4:
        Fz_z = np.gradient(field[2], axis=2)
        # Taking gradients of center layer only
        div = np.stack([Fx_x, Fy_y, Fz_z], axis=0)[:,:,:,1]
    else:                    
        div = np.stack([Fx_x, Fy_y], axis=0)
    
    return div.sum(axis=0)

def train_magnet(
    epochs: int = 101, betas: tuple = (1e-4, 0.02), n_T: int = 1000, features: int = 64, lr: float = 2e-4,
    batch_size: int = 128,
    device: str = "cuda:0", load_pth: Optional[str] = None
) -> None:
    
    cfg = {"epochs": epochs, "betas": betas, "n_T": n_T, 
                           "features":features, "lr":lr, "batch_size":batch_size }

    wandb.init(
        # set the wandb project where this run will be logged
        entity="dl4mag",
        project="mag-diffusion",

        # track hyperparameters and run metadata
        config=cfg
    )
    
    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=cfg["features"]), betas=cfg["betas"], n_T=cfg["n_T"])
    ddpm.to(device)

    magnetdb = h5py.File('/home/s214435/data/magfield_64_30000.h5')
    dbstd = np.std(magnetdb['field'])

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (dbstd, dbstd, dbstd))]
    )

    dataset = MagnetismData(
    magnetdb,
    transform=tf
    )

    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=20)
    optim = torch.optim.Adam(ddpm.parameters(), lr=cfg["lr"])

    for i in range(cfg["epochs"]):
        print(f"Epoch {i} : ")
        ddpm.train()


        pbar = tqdm(dataloader)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        if (i % 5 == 0):
            ddpm.eval()
            with torch.no_grad():
                xh = ddpm.sample(4, (3, 64, 64), device)
                fig, axes = plt.subplots(nrows=4, ncols=3, sharex=True,
                                        sharey=True, figsize=(8,8))
                norm = colors.Normalize(vmin=-1, vmax=1)
                
                tot_curl = 0
                tot_div = 0

                for j, sam in enumerate(xh):
                    sam_field = sam.cpu()
                    tot_curl = tot_curl + abs(curl(sam_field.numpy())).mean()
                    tot_div = tot_div + abs(div(sam_field.numpy())).mean()

                    for k, comp in enumerate(sam_field):
                        #img = comp.permute(1,2,0)
                        ax = axes.flat[j*3+k]
                        im = ax.imshow(comp.numpy(), cmap='bwr', norm=norm, origin="lower")

                cbar_ax = fig.add_axes([0.9, 0.345, 0.015, 0.3])
                fig.colorbar(im, cax=cbar_ax)
                fig.savefig(f"./contents/ddpm_sample_{i}.png")

                wandb.log({"loss": loss_ema, "avg_curl":tot_curl/4, "avg_div": tot_div/4, "sample":wandb.Image(fig)})

                plt.close()

                # save model
                torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")
        else:
            wandb.log({"loss": loss_ema})

    wandb.finish()


if __name__ == "__main__":
    train_magnet(epochs= 101, betas= (1e-4, 0.02), n_T= 1000, 
                           features=128, lr=2e-4, batch_size=192 )
