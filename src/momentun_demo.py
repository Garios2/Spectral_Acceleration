import torch
import matplotlib.pyplot as plt
from os import environ

dataset = "cifar10-5k"
arch = "fc-tanh"
loss = "mse"
gd_lr = 0.08
eig_freq = 100
cmap = plt.get_cmap('viridis')
gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/polyak/grad_search"
plt.figure(figsize=(5, 5), dpi=100)
for i in [1,2,3,4,5]:
    
    gd_train_loss = torch.load(f"{gd_directory}/train_loss_seed_is_{i}_step")
    sharpness = torch.load(f"{gd_directory}/eigs_seed_is_{i}_step")[:,0]

    plt.subplot(2, 1, 1)
    plt.plot(gd_train_loss,color=cmap(i/5))
    #plt.yscale('log') 
    plt.title("train loss")

    plt.subplot(2, 1, 2)
    plt.plot(torch.arange(len(sharpness))*eig_freq, sharpness, '-', color=cmap(i/5))
    plt.axhline(3.8 / gd_lr, linestyle='dotted')

plt.title("sharpness")
plt.xlabel("iteration")

plt.savefig(f"{gd_directory}/figures/momentun1,png", bbox_inches='tight', pad_inches=0)
