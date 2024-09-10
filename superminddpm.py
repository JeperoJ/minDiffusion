"""
Extremely Minimalistic Implementation of DDPM

https://arxiv.org/abs/2006.11239

Everything is self contained. (Except for pytorch and torchvision... of course)

run it with `python superminddpm.py`
"""

from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import h5py
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import wandb

from mindiffusion.unet import NaiveUnet


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)


class DummyEpsModel(nn.Module):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.conv = nn.Sequential(  # with batchnorm
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )

    def forward(self, x, t) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        return self.conv(x)


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x.device)  
        # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i
    
class MagnetismData(Dataset):
    def __init__(self, db, transform=None):
        self.db = db
        self.transform = transform

    def __len__(self):
        return len(self.db['field'][:][0])

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

    if field.shape[0]== 3:
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

    if field.shape[0] == 3:
        Fz_z = np.gradient(field[2], axis=2)
        # Taking gradients of center layer only
        div = np.stack([Fx_x, Fy_y, Fz_z], axis=0)[:,:,:,1]
    else:                    
        div = np.stack([Fx_x, Fy_y], axis=0)
    
    return div.sum(axis=0)

def train_mnist(n_epoch: int = 100, device="cuda:1") -> None:

    cfg = {"epochs": n_epoch, "betas": (1e-4, 0.02), "n_T": 1000, "features":16, "lr":2e-4, "batch_size":128 }
    
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="mag-diffusion",

        # track hyperparameters and run metadata
        config=cfg
    )
    
    ddpm = DDPM(eps_model=NaiveUnet(3, 3, cfg["features"]), betas=cfg["betas"], n_T=cfg["n_T"])
    ddpm.to(device)

    magnetdb = h5py.File('/home/s214435/data/magfield_64.h5')
    #dbmean = np.mean(magnetdb['field'][:][0])
    dbstd = np.std(magnetdb['field'])
    #print(dbstd)
    #print(magnetdb["field"].shape)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (dbstd, dbstd, dbstd))]
    )

    dataset = MagnetismData(
    magnetdb,
    transform=tf
    )

    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=20)
    optim = torch.optim.Adam(ddpm.parameters(), lr=cfg["lr"])

    for i in range(n_epoch):
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
                print("Hello")
                fig, axes = plt.subplots(nrows=4, ncols=3, sharex=True,
                                        sharey=True, figsize=(15,10))
                norm = colors.Normalize(vmin=-1, vmax=1)
                
                tot_curl = 0
                tot_div = 0

                for j, sam in enumerate(xh):
                    sam_field = sam.cpu()
                    tot_curl = tot_curl + abs(curl(sam_field.numpy())).mean()
                    tot_div = tot_div + abs(div(sam_field.numpy())).mean()

                    for k, comp in enumerate(sam_field):
                        img = comp.permute(1,2,0)
                        ax = axes.flat[j*3+k]
                        im = ax.imshow(img.numpy(), cmap='bwr', norm=norm, origin="lower")

                cbar_ax = fig.add_axes([0.9, 0.345, 0.015, 0.3])
                fig.colorbar(im, cax=cbar_ax)
                fig.savefig(f"./contents/ddpm_sample_{i}.png")
                plt.close()

                wandb.log({"loss": loss_ema, "avg_curl":tot_curl/4, "avg_div": tot_div/4})



                #grid = make_grid(xh, nrow=4)
                
                
                #save_image(grid, f"./contents/ddpm_sample_{i}.png")

                # save model
                torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")

    wandb.finish()


if __name__ == "__main__":
    train_mnist()
