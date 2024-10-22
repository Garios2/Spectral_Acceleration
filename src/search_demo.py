import torch
import matplotlib.pyplot as plt
from os import environ
import numpy as np
from matplotlib.gridspec import GridSpec

import os
dataset = "cifar10-5k"
arch = "cnn-relu"
loss = "mse"
gd_lr = 0.01
gd_eig_freq = 100
scaling = 1.5
cmap = plt.get_cmap('viridis')
results_dir = os.environ["RESULTS"]
directory = f"{results_dir}/{dataset}/{arch}/seed_0/{loss}/gd/grad_search"
dir = f"{directory}/figures"
if not os.path.exists(dir):
    os.makedirs(dir)


fig = plt.figure(figsize=(10, 10), dpi=100)
gs = GridSpec(2, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
cmap = plt.get_cmap('viridis')
end_time=[]

lr = 0.0505
    
train_loss = torch.load(f"{directory}/train_loss_learning_rate_{lr}_step")
test_loss = torch.load(f"{directory}/test_loss_learning_rate_{lr}_step")
print(len(train_loss))
ax1.plot(train_loss)
ax2.plot(test_loss)

ax2.set_xlabel("iteration")
ax2.set_ylabel("test loss")
ax1.set_ylabel("train loss")
#ax1.set_yscale('log')
ax1.set_xlabel("iteration")
fig_save = f"{directory}/figures/grad_search.svg"
plt.savefig(fig_save, bbox_inches='tight', pad_inches=0)