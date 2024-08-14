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
gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"

mode='global_scaling'
scaling=1.0
nfilter=20
save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
gd_train_loss = torch.load(f"{gd_directory}/train_loss_{save_name}")

mode='flat_scaling_v2'

if mode != 'original':
    scaling=1.5
    nfilter=20
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_{save_name}")
    train_loss_flat = torch.cat((torch.tensor([np.nan]*3000), gd_train_loss_flat))


plt.figure(figsize=(10, 10), dpi=100)

plt.subplot(2, 1, 1) 
plt.plot(gd_train_loss)
if mode != 'original':
    plt.plot(train_loss_flat, color = "green")
#plt.plot(train_loss_global, color = "red")
#plt.plot(train_loss_flat, color = "green")
#plt.yscale('log') 
plt.title("train loss")
#plt.axvline(6000, linestyle='dotted')



plt.subplot(2, 1, 2) 


for i in range(20):
    if mode != 'original':
        pre_sharpness = torch.load(f"{gd_directory}/eigs_global_scaling_1.0_top_20_step")[:30,i]
    else:
        pre_sharpness = torch.load(f"{gd_directory}/eigs_global_scaling_1.0_top_20_step")[:,i]

    if mode != 'original':
        gd_sharpness = torch.cat((torch.tensor([np.nan]*30), torch.load(f"{gd_directory}/eigs_{save_name}")[:,i]))
    if i == 9:
        plt.plot(torch.arange(len(pre_sharpness)) * gd_eig_freq, pre_sharpness, color="red",label="10th largerst elgenval")
        if mode != 'original':
            plt.plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="red",label="10th largerst elgenval")

    else:
        plt.plot(torch.arange(len(pre_sharpness)) * gd_eig_freq, pre_sharpness, color=cmap(i/20))
        if mode != 'original':
            plt.plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=cmap(i/20))

if mode != 'original':
    plt.axvline(3000, linestyle='dotted',color='red')
plt.axhline(2. / gd_lr, linestyle='dotted')
plt.axhline(2. / (gd_lr*scaling), linestyle='dotted')
#plt.axvline(6000, linestyle='dotted')
plt.title("sharpness")
plt.legend()
plt.xlabel("iteration")
makedirs(f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}/figures", exist_ok=True)
plt.savefig(f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}/figures/sharpness_flow_{mode}_{scaling}.png", bbox_inches='tight', pad_inches=0)
