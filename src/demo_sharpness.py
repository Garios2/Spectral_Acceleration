import torch
import matplotlib.pyplot as plt
from os import environ,makedirs
import numpy as np

dataset = "cifar10-20k"
arch = "fc-tanh"
loss = "mse"
gd_lr = 0.05
gd_eig_freq = 100
cmap = plt.get_cmap('viridis')
scaling = 1.0
gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"

gd_train_loss = torch.load(f"{gd_directory}/train_loss_0k-10k-global_scaling-{scaling}_step")
gd_train_acc = torch.load(f"{gd_directory}/train_acc_0k-10k-global_scaling-{scaling}_step")




"""
gd_train_loss_global = torch.load(f"{gd_directory}/train_loss_6k-10k-global-scaling-2.5_step")
gd_train_acc_global = torch.load(f"{gd_directory}/train_acc_6k-10k-global-scaling-2.5_step")
gd_sharpness_global = torch.load(f"{gd_directory}/eigs_6k-10k-global-scaling-2.5_step")[:,0]
train_loss_global = torch.cat((torch.tensor([np.nan]*6000), gd_train_loss_global))
train_acc_global = torch.cat((torch.tensor([np.nan]*6000), gd_train_acc_global))

gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_6k-10k-flat-scaling-2.5_step")
gd_train_acc_flat = torch.load(f"{gd_directory}/train_acc_6k-10k-flat-scaling-2.5_step")
gd_sharpness_flat = torch.load(f"{gd_directory}/eigs_6k-10k-flat-scaling-2.5_step")[:,0]
train_loss_flat = torch.cat((torch.tensor([np.nan]*6000), gd_train_loss_flat))
train_acc_flat = torch.cat((torch.tensor([np.nan]*6000), gd_train_acc_flat))
"""





plt.figure(figsize=(10, 10), dpi=100)

plt.subplot(2, 1, 1) 
plt.plot(gd_train_loss)
#plt.plot(train_loss_global, color = "red")
#plt.plot(train_loss_flat, color = "green")
#plt.yscale('log') 
plt.title("train loss")
#plt.axvline(6000, linestyle='dotted')



#plt.scatter(6000+torch.arange(len(gd_sharpness_global)) * gd_eig_freq, gd_sharpness_global, s=5, color="red")

#for i in range(9):
    #pre_sharpness = torch.load(f"{gd_directory}/eigs_global-scaling-1_step")
    #pre_sharpness = torch.load(f"{gd_directory}/eigs_final")[:6000,i]
    #gd_sharpness = torch.load(f"{gd_directory}/eigs_6k-10k-global-scaling-1.5_step")[:,i]

    #plt.scatter(6000+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, s=5,color=cmap(i/9))
    #plt.scatter(torch.arange(len(pre_sharpness)) * gd_eig_freq, pre_sharpness, s=5,color=cmap(i/9))

plt.subplot(2, 1, 2) 

#plt.scatter(6000+torch.arange(len(gd_sharpness_global)) * gd_eig_freq, gd_sharpness_global, s=5, color="red")

for i in range(20):
    pre_sharpness = torch.load(f"{gd_directory}/eigs_0k-10k-global_scaling-{scaling}_step")[:,i]
    #pre_sharpness = torch.load(f"{gd_directory}/eigs_final")[:6000,i]
    #gd_sharpness = torch.load(f"{gd_directory}/eigs_6k-10k-flat-scaling-1.5_step")[:,i]
    if i == 9:
        plt.plot(torch.arange(len(pre_sharpness)) * gd_eig_freq, pre_sharpness, color="red",label="10th largerst elgenval")
    else:
        plt.plot(torch.arange(len(pre_sharpness)) * gd_eig_freq, pre_sharpness, color=cmap(i/20))
    #plt.scatter(6000+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, s=5,color=cmap(i/9))
    #plt.scatter(torch.arange(len(pre_sharpness)) * gd_eig_freq, pre_sharpness, s=5,color=cmap(i/9))


plt.axhline(2. / gd_lr, linestyle='dotted')
#plt.axvline(6000, linestyle='dotted')
plt.title("sharpness")
plt.legend()
plt.xlabel("iteration")
makedirs(f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}/figures", exist_ok=True)
plt.savefig(f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}/figures/sharpness_flow_1.png", bbox_inches='tight', pad_inches=0)
