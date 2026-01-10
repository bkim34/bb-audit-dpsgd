import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.autograd import Variable

from torchvision import datasets
from torchvision.transforms import transforms

#from mine.models.gan import GAN

#from mine.datasets import FunctionDataset, MultivariateNormalDataset
from layers import ConcatLayer, CustomSequential

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import utils

torch.autograd.set_detect_anomaly(False)

EPS = 1e-6

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print("Device:", device)


EPS = 1e-6

class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, running_ema):
        # x: (b,) or (b,1) flattened later
        x = x.reshape(-1)
        ctx.save_for_backward(x, running_ema)

        # stable log(mean(exp(x)))
        return torch.logsumexp(x, dim=0) - math.log(x.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        x, running_mean = ctx.saved_tensors
        x = x.reshape(-1)

        # more stable gradient than exp(x) directly:
        m = x.max().detach()
        exp_shift = torch.exp(x - m).detach()

        # exp(-m) * running_mean ~= mean(exp(x-m))
        denom = (running_mean * torch.exp(-m) + EPS) * x.shape[0]

        grad = grad_output * exp_shift / denom
        return grad, None


def logmeanexp(x, dim=0):
    return torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])


def ema_loss(x, running_mean, ema_rate):
    """
    x: tensor (b,) or (b,1)
    running_mean: scalar tensor buffer
    ema_rate: beta in the paper
    """
    x = x.reshape(-1)

    # exp(mean) in a stable way (this is mean(exp(x)))
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()

    # running_mean is a scalar tensor buffer
    if running_mean.item() == 0.0:
        running_mean = t_exp
    else:
        running_mean = ema_rate * t_exp + (1.0 - ema_rate) * running_mean

    t_log = EMALoss.apply(x, running_mean)
    return t_log, running_mean

class Renyi(nn.Module):
    def __init__(self, T_net, renyi_order, ema_rate=0.01):
        super().__init__()
        if renyi_order is None or renyi_order <= 1:
            raise ValueError("renyi_order must be > 1")

        self.T = T_net
        self.renyi_order = float(renyi_order)
        self.ema_rate = float(ema_rate)

        self.register_buffer("running_mean_q", torch.tensor(0.0))
        self.register_buffer("running_mean_p", torch.tensor(0.0))

    def forward(self, q_batch, p_batch, update_ema=True):
        a = self.renyi_order
        t_q = self.T(q_batch).reshape(-1)
        t_p = self.T(p_batch).reshape(-1)

        if update_ema:
            log_mq, m_q = ema_loss((a - 1.0) * t_q, self.running_mean_q, self.ema_rate)
            log_mp, m_p = ema_loss(a * t_p,         self.running_mean_p, self.ema_rate)
            self.running_mean_q.copy_(m_q)
            self.running_mean_p.copy_(m_p)
        else:
            log_mq = logmeanexp((a - 1.0) * t_q, dim=0)
            log_mp = logmeanexp(a * t_p,         dim=0)

        renyi_lb = (log_mq / (a - 1.0)) - (log_mp / a)
        return -renyi_lb


    def estimate(self, Q, P):
        if isinstance(Q, np.ndarray): Q = torch.from_numpy(Q).float()
        if isinstance(P, np.ndarray): P = torch.from_numpy(P).float()

        device = next(self.parameters()).device
        Q, P = Q.to(device), P.to(device)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            est = -self.forward(Q, P, update_ema=False)
        if was_training:
            self.train()
        return est

    def optimize(self, Q, P, iters, batch_size, opt=None, lr=1e-4):
        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        device = next(self.parameters()).device

        for it in range(1, iters + 1):
            avg_bound = 0.0
            n_steps = 0

            # THIS is where your two lines go (see next section)
            for q_b, p_b in utils.batch(Q, P, batch_size):
                if isinstance(q_b, np.ndarray): q_b = torch.from_numpy(q_b).float()
                if isinstance(p_b, np.ndarray): p_b = torch.from_numpy(p_b).float()
                q_b, p_b = q_b.to(device), p_b.to(device)

                opt.zero_grad()
                loss = self.forward(q_b, p_b, update_ema=True)
                loss.backward()
                opt.step()

                avg_bound += (-loss.item())
                n_steps += 1

            if it % max(1, iters // 3) == 0:
                print(f"Iter {it} - avg bound: {avg_bound / max(1, n_steps)}")

        final_est = self.estimate(Q, P)
        print(f"Final estimate: {final_est}")
        return final_est

class T(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (b,1) or (b,)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x)  # (b,1)



# class T(nn.Module):
#     def __init__(self, x_dim, z_dim):
#         super().__init__()
#         self.layers = CustomSequential(ConcatLayer(), nn.Linear(x_dim, 100),
#                                        nn.ReLU(),
#                                        nn.Linear(100, 100),
#                                        nn.ReLU(),
#                                        nn.Linear(100, 1))

#     def forward(self, x):
#         # x: (b,1) or (b,)
#         if x.dim() == 1:
#             x = x.unsqueeze(-1)
        # return self.net(x)  # (b,1)

class PairedQP(torch.utils.data.Dataset):
    def __init__(self, Q, P):
        self.Q = torch.as_tensor(Q, dtype=torch.float32).view(-1, 1)
        self.P = torch.as_tensor(P, dtype=torch.float32).view(-1, 1)
        self.n = min(len(self.Q), len(self.P))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.Q[i], self.P[i]  # <-- TWO tensors


class RenyiDivergenceEstimator(pl.LightningModule):
    def __init__(self, lr=1e-4, renyi_order=1.1, ema_rate=0.98, hidden=64):
        super().__init__()
        self.save_hyperparameters()  # makes hparams visible in logs/checkpoints

        self.T = T(hidden=hidden)
        self.energy_loss = Renyi(self.T, renyi_order=renyi_order, ema_rate=ema_rate)

        # optional convenience attribute (we'll fill it after test)
        self.avg_test_renyi_lb = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        q, p = batch

        loss = self.energy_loss(q, p, update_ema=True)
        renyi_lb = -loss

        self.log_dict(
            {"train_loss": loss, "train_renyi_lb": renyi_lb},
            on_step=True,
            on_epoch=True,
            prog_bar=True,   # this is what makes it appear on the bar :contentReference[oaicite:4]{index=4}
            logger=True,
            add_dataloader_idx=False,
        )

        return loss


    def test_step(self, batch, batch_idx):
        q, p = batch

        loss = self.energy_loss(q, p, update_ema=False)
        renyi_lb = -loss

        # Log BOTH to the progress bar and to the returned results dict
        self.log_dict(
            {"test_loss": loss, "test_renyi_lb": renyi_lb},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,  # IMPORTANT: prevents /dataloader_idx_0 suffix :contentReference[oaicite:1]{index=1}
        )

        return {"test_renyi_lb": renyi_lb}


    def on_test_epoch_end(self):
        # `test_renyi_lb` logged with on_epoch=True is now aggregated (mean by default)
        v = self.trainer.callback_metrics.get("test_renyi_lb")
        if v is not None:
            self.avg_test_renyi_lb = float(v.detach().cpu().item())



#if __name__ == '__main__':
    
    # function_experiment()
    # gan_experiment()
