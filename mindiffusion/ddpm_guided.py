from typing import Dict, Optional, Tuple


import torch
import torch.nn as nn
#from torch.utils.data import Dataset, DataLoader

#Based on: https://arxiv.org/pdf/2006.11239 (p1)
#Modified as in: https://arxiv.org/pdf/2406.17763 (p2) 

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

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device)
        # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def sample(self, n_sample: int, size, device, guide_weight: float = 0.0, guide_fun: Optional[function] = None) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device).requires_grad_(True)  # x_T ~ N(0, 1)
        if guide_fun == None:
            with torch.no_grad():
                # This samples accordingly to Algorithm 2 in p1
                for i in range(self.n_T, 0, -1):
                    z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                    eps = self.eps_model(
                        x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1)
                    )

                    x_i = (
                        self.oneover_sqrta[i] * (x_i - eps  * self.mab_over_sqrtmab[i])
                        + self.sqrt_beta_t[i] * z
                    )
        else:
            #Samples according to Algorithm 2 in p1, modified with the guidance from p2
            for i in range(self.n_T, 0, -1):
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                eps = self.eps_model(
                    x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1)
                )
                l_guide = torch.sum(torch.vmap(guide_fun)(x_i-eps))
                l_guide.backward()

                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps  * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z - guide_weight * x_i.grad
                ).detach().requires_grad_(True)

        return x_i.detach()


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
