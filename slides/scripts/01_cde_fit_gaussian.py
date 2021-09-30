import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch import tensor
from torch.distributions import Normal
from math import pi, log
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(0)


if not os.path.exists("figures/2_1_cde"):
    os.makedirs("figures/2_1_cde")


mean = 1.0
std = 0.4
normal_dist = Normal(mean, std)      # We do not usually know this...
samples = normal_dist.sample((50,))  # ...but all we have are these samples.

with mpl.rc_context(fname="../.matplotlibrc"):
    fig = plt.figure(figsize=(6, 2))
    plt.plot(samples, np.zeros_like(samples), 'bx',
             alpha=0.5, markerfacecolor='none', markersize=6)
    plt.xlim([-4, 4])
    plt.ylim([-0.12, 1.2])
    plt.xlabel("x")
    plt.savefig("figures/2_1_cde/samples.png", dpi=150, bbox_inches="tight")


# Fitting a Gaussian to data with maximum likelihood
dataset = data.TensorDataset(samples)
train_loader = data.DataLoader(samples, batch_size=10)
learned_mean = torch.nn.Parameter(torch.zeros(1))
learned_log_std = torch.nn.Parameter(torch.zeros(1))
opt = optim.Adam([learned_mean, learned_log_std], lr=0.005)

for e in range(500):
    for sample_batch in train_loader:
        opt.zero_grad()
        learned_std = torch.exp(learned_log_std)
        prob_Gauss = (1/torch.sqrt(2*pi*learned_std**2) *
                      torch.exp(-0.5/learned_std**2 * (sample_batch-learned_mean)**2))
        loss = -torch.log(prob_Gauss).sum()
        loss.backward()
        opt.step()

print(
    f"""Learned mean: {learned_mean.item()},
learned standard deviation: {torch.exp(learned_log_std).item()}""")


true_dist = Normal(mean, std)
learned_dist = Normal(learned_mean, torch.exp(learned_log_std))

x = torch.linspace(-4, 4, 100)
true_probs = torch.exp(normal_dist.log_prob(x))
learned_probs = torch.exp(learned_dist.log_prob(x)).detach()


with mpl.rc_context(fname="../.matplotlibrc"):
    fig = plt.figure(figsize=(6, 2))
    plt.plot(samples, np.zeros_like(samples), 'bx',
             alpha=0.5, markerfacecolor='none', markersize=6)
    plt.plot(x, true_probs)
    plt.plot(x, learned_probs)
    plt.legend(["Samples", "Ground truth", "Learned"], loc="upper left")
    plt.xlim([-4, 4])
    plt.ylim([-0.12, 1.2])
    plt.xlabel("x")
    plt.savefig("figures/2_1_cde/fitted_samples.png",
                dpi=150, bbox_inches="tight")
