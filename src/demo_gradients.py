import torch
import matplotlib.pyplot as plt
from os import environ,makedirs
import numpy as np

dataset = "cifar10"
arch = "fc-tanh"
loss = "mse"
gd_lr = 0.04
gd_eig_freq = 100
cmap = plt.get_cmap('viridis')
scaling = 1.0
momentum = "polyak"
middlePoint=1000
gdm_lr = gd_lr/10
compare_flat_scaling = 0
compare_gdm_beta_99 =0
compare_gdm_beta_50 =0

compare_bulk_gd_1 = 1
compare_bulk_gd_3 = 0
compare_gd = 1
compare_gdm=0
compare_gdm_warmup_beta=0
#gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"
fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=100, sharex=True)
gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"
gdm_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{gdm_lr}_beta_0.9"







axs[0].set_title("train loss")


if compare_gd == 1:
    color = "green"
    mode='global_scaling'
    scaling=1.0
    nfilter=1
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss = torch.load(f"{gd_directory}/train_loss_{save_name}")
    gradient = torch.load(f"{gd_directory}/grads_{save_name}")
    g_norm = [torch.norm(u) for u in gradient]
    axs[0].plot(gd_train_loss,label=f"GD,eta={gd_lr}", color=color)
    axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)

if compare_bulk_gd_1 == 1:
    color = "red"
    mode='flat_scaling_v1'
    scaling=1.0
    nfilter=60
    save_name = "bulk_{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss = torch.load(f"{gd_directory}/train_loss_{save_name}")
    train_loss = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss))
    gradient = torch.load(f"{gd_directory}/grads_{save_name}")
    g_norm = [torch.norm(u) for u in gradient]
    nan_list = [np.nan] * int(middlePoint/gd_eig_freq)
    g_norm = nan_list + g_norm

    axs[0].plot(train_loss,label=f"Bulk_GD,eta={gd_lr}", color=color)
    axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)

    filtered_gradient = torch.load(f"{gd_directory}/filtered_grads_{save_name}")
    g_norm = [torch.norm(u) for u in filtered_gradient]
    nan_list = [np.nan] * int(middlePoint/gd_eig_freq)
    g_norm = nan_list + g_norm
    axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color="blue", label = "filtered_grads")

if compare_bulk_gd_3 == 1:
    color = "blue"
    mode='flat_scaling_v1'
    scaling=3.0
    nfilter=80
    save_name = "bulk_{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss = torch.load(f"{gd_directory}/train_loss_{save_name}")
    train_loss = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss))
    gradient = torch.load(f"{gd_directory}/grads_{save_name}")
    g_norm = [torch.norm(u) for u in gradient]
    nan_list = [np.nan] * int(middlePoint/gd_eig_freq)
    g_norm = nan_list + g_norm
    axs[0].plot(train_loss,label=f"Bulk_GD_3.0,eta={gd_lr}", color=color)
    axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)

if compare_flat_scaling==1:
    color = "purple"
    mode_flat='flat_scaling_v1'
    scaling_flat=1.5
    nfilter_flat=20
    save_name = "{}_{}_top_{}_step".format(mode_flat, scaling_flat,nfilter_flat)
    gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_{save_name}")
    train_loss_flat = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_flat))
    gradient = torch.load(f"{gd_directory}/grads_{save_name}")
    g_norm = [torch.norm(u) for u in gradient]
    nan_list = [np.nan] * int(middlePoint/gd_eig_freq)
    g_norm = nan_list + g_norm
    axs[0].plot(train_loss_flat, color = color,label=f"GD,flat_scaling_{scaling_flat}")
    axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)
 
    color = "yellow"
    mode_flat='flat_scaling_v2'
    scaling_flat=1.5
    nfilter_flat=20
    save_name = "{}_{}_top_{}_step".format(mode_flat, scaling_flat,nfilter_flat)
    gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_{save_name}")
    train_loss_flat = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_flat))
    gradient = torch.load(f"{gd_directory}/grads_{save_name}")
    g_norm = [torch.norm(u) for u in gradient]
    nan_list = [np.nan] * int(middlePoint/gd_eig_freq)
    g_norm = nan_list + g_norm
    axs[0].plot(train_loss_flat, color = color,label=f"GD,flat_scaling_{scaling_flat}")
    axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)

if compare_gdm ==1:
    color = "purple"
    mode='global_scaling'
    scaling=1.0
    nfilter=1
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gdm_train_loss = torch.load(f"{gdm_directory}/train_loss_{save_name}")
    gradient = torch.load(f"{gdm_directory}/grads_{save_name}")
    g_norm = [torch.norm(u) for u in gradient]
    axs[0].plot(gdm_train_loss,label=f"GDM,beta=0.9", color=color)
    axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)

if compare_gdm_warmup_beta==1:
    color = "purple"
    lr = gd_lr/10
    gdm_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{lr}_beta_0.9/warmup_beta_0.99"
    mode='global_scaling'
    scaling=1.0
    nfilter=1
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)

    gdm_train_loss = torch.load(f"{gdm_directory}/train_loss_{save_name}")
    gradient = torch.load(f"{gdm_directory}/grads_{save_name}")
    g_norm = [torch.norm(u) for u in gradient]
    axs[0].plot(gdm_train_loss,label=f"GDM,warmup beta", color=color)
    axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)


if compare_gdm_beta_99==1:
    color = "green"
    gdm_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{gdm_lr/10}_beta_0.99"
    mode='global_scaling'
    scaling=1.0
    nfilter=20
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gdm_train_loss = torch.load(f"{gdm_directory}/train_loss_{save_name}")
    axs[0].plot(gdm_train_loss,label=f"GDM,beta=0.99", color=color)
    
    for i in range(5):
        gdm_sharpness = torch.load(f"{gdm_directory}/eigs_{save_name}")[:,i]
        axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color=color)
        if i== 4:
            axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color=color,label="beta0.99")
    axs[1].axhline(2*(1.9) / gdm_lr, linestyle='dotted',label="MSS_GDM")

if compare_gdm_beta_50==1:
    color = "blue"
    lr = gd_lr/2
    gdm_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{lr}_beta_0.5"
    mode='global_scaling'
    scaling=1.0
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    nfilter=20
    gdm_train_loss = torch.load(f"{gdm_directory}/train_loss_{save_name}")
    axs[0].plot(gdm_train_loss,label=f"GDM,beta=0.5", color=color)

    for i in range(5):
        gdm_sharpness = torch.load(f"{gdm_directory}/eigs_{save_name}")[:,i]
        axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color=color)
        if i== 4:
            axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color=color,label="beta0.5")
    axs[1].axhline(2*(1.9) / lr, linestyle='dotted',label="MSS_GDM")





axs[0].legend()

#axs[1].set_xlim(0,2500)


axs[1].set_title("gradient norm")
axs[1].legend()
axs[1].set_xlabel("iteration")



makedirs(f"{gd_directory}/figures", exist_ok=True)
#plt.show()
plt.savefig(f"{gd_directory}/figures/gradientnorm_GDM_cifar10_1.png", bbox_inches='tight', pad_inches=0)

