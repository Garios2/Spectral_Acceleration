import torch
import matplotlib.pyplot as plt
from os import environ

dataset = "cifar10-5k"
arch = "fc-tanh"
loss = "mse"
gd_lr = 0.01
gd_eig_freq = 100

gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"

gd_train_loss = torch.load(f"{gd_directory}/train_loss_0-6k_step")
gd_train_acc = torch.load(f"{gd_directory}/train_acc_0-6k_step")

gd_gradients = torch.load(f"{gd_directory}/grads_0-6k_step")



plt.figure(figsize=(5, 5), dpi=100)

plt.subplot(3, 1, 1)
plt.plot(gd_train_loss)
plt.yscale('log') 
plt.title("train loss")

cmap = plt.get_cmap('viridis')
plt.subplot(3, 1, 2)
for i in range(9):
    gd_eigendir = torch.load(f"{gd_directory}/eigvecs_0-6k_step")[:,:,i]
    content = []
    for j in range(len(gd_gradients)):
        content.append(torch.square(torch.dot(gd_gradients[j], gd_eigendir[j]) ))
    plt.plot(content, color = cmap(i/9))


plt.title("gradient")

plt.subplot(3, 1, 3)
for i in range(9):
    gd_sharpness = torch.load(f"{gd_directory}/eigs_0-6k_step")[:,i]
    plt.scatter(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, s=5,color=cmap(i/9))


plt.axhline(2. / gd_lr, linestyle='dotted')
plt.title("sharpness")
plt.xlabel("iteration")

plt.savefig(f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}/0-6k_gradients", bbox_inches='tight', pad_inches=0)
