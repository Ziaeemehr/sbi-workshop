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


n = 4000
d = 1
theta = torch.rand((n, d))
noise = torch.randn((n, d)) * 0.05
x = theta + 0.3 * torch.sin(2*pi*theta) + noise


# Fitting a Gaussian to data with maximum likelihood
dataset = data.TensorDataset(theta, x)
train_loader = data.DataLoader(dataset, batch_size=30)


net = nn.Sequential(
    nn.Linear(1, 20), nn.ReLU(),
    nn.Linear(20, 20), nn.ReLU(),
    nn.Linear(20, 2)
)

opt = optim.Adam(net.parameters(), lr=0.01)

for e in range(100):
    for theta_batch, x_batch in train_loader:
        opt.zero_grad()
        nn_output = net(theta_batch)
        mean = nn_output[:, 0].unsqueeze(1)
        std = torch.exp(nn_output[:, 1]).unsqueeze(1)
        prob_Gauss = (1/torch.sqrt(2*pi*std**2) *
                      torch.exp(-.5/std**2*(mean-x_batch)**2))
        loss = -torch.log(prob_Gauss).sum()
        loss.backward()
        opt.step()


# the theta at which we evaluate the neural network.
theta_test = tensor([0.1])
nn_output = net(theta_test)
conditional_mean = nn_output[0].detach().numpy()
conditional_std = torch.exp(nn_output[1]).detach().numpy()
print("Conditional mean: ", conditional_mean)
print("Conditional std: ", conditional_std)

# Data t_train and x_train
with mpl.rc_context(fname="../.matplotlibrc"):

    fig = plt.figure(figsize=(4.5, 2.2))
    plt.plot(theta[:400], x[:400], 'go', alpha=0.4,
             markerfacecolor='none', zorder=-100)
    plt.plot([theta_test, theta_test], [conditional_mean-conditional_std,
             conditional_mean+conditional_std], c="r", linewidth=2)
    plt.scatter(theta_test, conditional_mean, c="r", s=30, alpha=1.0)
    plt.xlabel(r'$\theta$')
    plt.ylabel('x')
    plt.savefig("figures/2_1_cde/cde_fitted.png", dpi=200, bbox_inches="tight")

samples = []
theta_test = torch.linspace(0.0, 1.0, 500).unsqueeze(1)

for single_theta in theta_test:
    network_outs = net(single_theta.unsqueeze(1))
    m = network_outs[:, 0]
    s = torch.exp(network_outs[:, 1])
    conditional_distribution = Normal(m, s)
    sample = conditional_distribution.sample((1,))
    samples.append(sample)
samples = torch.stack(samples)
