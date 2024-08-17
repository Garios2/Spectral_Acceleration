import torch
import matplotlib.pyplot as plt
from os import environ,makedirs
import numpy as np

dataset = "cifar10"
arch = "fc-tanh"
loss = "mse"
gd_lr = 0.02
gd_eig_freq = 100
cmap = plt.get_cmap('viridis')
scaling = 1.0
momentum = "polyak"
middlePoint=2000

#gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"
gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"
if momentum == "polyak":
    gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{gd_lr}_beta_0.9"

mode='global_scaling'
scaling=1.0
nfilter=20
save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
gd_train_loss = torch.load(f"{gd_directory}/train_loss_{save_name}")
"""
original, flat_scaling_v1, flat_scaling_v2
"""
mode='flat_scaling_v2'

if mode != 'original':
    scaling=1.5
    nfilter=20
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_{save_name}")
    train_loss_flat = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_flat))


fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=100, sharex=True)

axs[0].plot(gd_train_loss)
if mode != 'original':
    axs[0].plot(train_loss_flat, color = "green")
axs[0].set_title("train loss")

for i in range(20):
    if mode != 'original':
        pre_sharpness = torch.load(f"{gd_directory}/eigs_global_scaling_1.0_top_20_step")[:int(middlePoint/gd_eig_freq),i]
    else:
        pre_sharpness = torch.load(f"{gd_directory}/eigs_global_scaling_1.0_top_20_step")[:,i]

    if mode != 'original':
        gd_sharpness = torch.cat((torch.tensor([np.nan]*(int(middlePoint/gd_eig_freq))), torch.load(f"{gd_directory}/eigs_{save_name}")[:,i]))
    if i == 9:
        axs[1].plot(torch.arange(len(pre_sharpness)) * gd_eig_freq, pre_sharpness, color="red",label="10th largerst elgenval")
        if mode != 'original':
            axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="red",label="10th largerst elgenval")

    else:
        axs[1].plot(torch.arange(len(pre_sharpness)) * gd_eig_freq, pre_sharpness, color=cmap(i/nfilter))
        if mode != 'original':
            axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=cmap(i/nfilter))

if nfilter>20:
    for i in range(nfilter-20):
        gd_sharpness = torch.cat((torch.tensor([np.nan]*30), torch.load(f"{gd_directory}/eigs_{save_name}")[:,i+20]))
        axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=cmap((20+i)/nfilter))




if mode != 'original':
    axs[1].axvline(middlePoint, linestyle='dotted',color='red')
if momentum == "polyak":
    axs[1].axhline(2*(1.9) / gd_lr, linestyle='dotted')
    axs[1].axhline(2*(1.9) / (gd_lr*scaling), linestyle='dotted')
else:
    axs[1].axhline(2. / gd_lr, linestyle='dotted')
    axs[1].axhline(2. / (gd_lr*scaling), linestyle='dotted')
axs[1].set_title("sharpness")
axs[1].legend()
axs[1].set_xlabel("iteration")

makedirs(f"{gd_directory}/figures", exist_ok=True)
plt.savefig(f"{gd_directory}/figures/sharpness_flow_{mode}_{scaling}_top{nfilter}.png", bbox_inches='tight', pad_inches=0)

