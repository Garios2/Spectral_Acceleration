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
compare_gd = 1
compare_gdm=1
compare_bulk_gd = 1
compare_gdm_another =1
#gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"
fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=100, sharex=True)
gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"
gdm_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{gdm_lr}_beta_0.9"







axs[0].set_title("train loss")


if compare_gd ==1:
    mode='global_scaling'
    scaling=1.0
    nfilter=20
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss = torch.load(f"{gd_directory}/train_loss_{save_name}")
    axs[0].plot(gd_train_loss,label=f"GD,eta={gd_lr}", color="MediumBlue")

    for i in range(5):
        gd_sharpness = torch.load(f"{gd_directory}/eigs_{save_name}")[:,i]
        axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="MediumBlue")
        if i== 4:
            axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="MediumBlue",label="GD_top5")
    axs[1].axhline(2. / (gd_lr), linestyle='dotted',label="MSS_GD")


if compare_gdm ==1:
    mode='global_scaling'
    scaling=1.0
    nfilter=20
    gdm_train_loss = torch.load(f"{gdm_directory}/train_loss_{save_name}")
    axs[0].plot(gdm_train_loss,label=f"GDM,eta={gdm_lr}", color="orange")
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    for i in range(5):
        gdm_sharpness = torch.load(f"{gdm_directory}/eigs_{save_name}")[:,i]
        axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color="orange")
        if i== 4:
            axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color="orange",label="GDM_top5")
    axs[1].axhline(2*(1.9) / gdm_lr, linestyle='dotted',label="MSS_GDM")

if compare_gdm_another ==1:
    gdm_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{gdm_lr/10}_beta_0.99"
    mode='global_scaling'
    scaling=1.0
    nfilter=20
    gdm_train_loss = torch.load(f"{gdm_directory}/train_loss_{save_name}")
    axs[0].plot(gdm_train_loss,label=f"GDM,eta={gdm_lr}", color="grey")
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    for i in range(5):
        gdm_sharpness = torch.load(f"{gdm_directory}/eigs_{save_name}")[:,i]
        axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color="grey")
        if i== 4:
            axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color="grey",label="GDM_top5")
    axs[1].axhline(2*(1.9) / gdm_lr, linestyle='dotted',label="MSS_GDM")

if compare_flat_scaling==1:
    mode_flat='flat_scaling_v1'
    scaling_flat=3.0
    nfilter_flat=40
    save_name_flat = "{}_{}_top_{}_step".format(mode_flat, scaling_flat,nfilter_flat)
    gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_{save_name_flat}")
    train_loss_flat = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_flat))
    axs[0].plot(train_loss_flat, color = "purple",label=f"GD,flat_scaling_{scaling_flat}")

    for i in range(5):
        gd_sharpness = torch.load(f"{gd_directory}/eigs_{save_name_flat}")[:,i]
        axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="purple")
        if i== 4:
            axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="purple",label=f"GD_{mode_flat}_{scaling_flat}_top5")

    mode_flat='flat_scaling_v1'
    scaling_flat=1.0
    nfilter_flat=20
    save_name_flat = "{}_{}_top_{}_step".format(mode_flat, scaling_flat,nfilter_flat)
    gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_{save_name_flat}")
    train_loss_flat = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_flat))
    axs[0].plot(train_loss_flat, color = "#4682B4",label=f"GD,flat_scaling_{scaling_flat}")

    for i in range(5):
        gd_sharpness = torch.load(f"{gd_directory}/eigs_{save_name_flat}")[:,i]
        axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="#4682B4")
        if i== 4:
            axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="#4682B4",label=f"GD_{mode_flat}_{scaling_flat}_top5")

if compare_bulk_gd ==1:
    mode_bulk='flat_scaling_v1'
    scaling_bulk=3.0
    nfilter_bulk=80
    save_name_bulk = "bulk_{}_{}_top_{}_step".format(mode_bulk, scaling_bulk,nfilter_bulk)
    gd_train_loss_bulk = torch.load(f"{gd_directory}/train_loss_{save_name_bulk}")
    train_loss_bulk = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_bulk))
    axs[0].plot(train_loss_bulk, color = "green",label=f"GD,bulk_scaling_{scaling_bulk}")

    for i in range(5):
        gd_sharpness = torch.load(f"{gd_directory}/eigs_{save_name_bulk}")[:,i]
        axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="green")
        if i== 4:
            axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="green",label=f"GD_{mode_bulk}_{scaling_bulk}_top5")

    mode_bulk='flat_scaling_v1'
    scaling_bulk=1.0
    nfilter_bulk=50
    save_name_bulk = "bulk_{}_{}_top_{}_step".format(mode_bulk, scaling_bulk,nfilter_bulk)
    gd_train_loss_bulk = torch.load(f"{gd_directory}/train_loss_{save_name_bulk}")
    train_loss_bulk = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_bulk))
    axs[0].plot(train_loss_bulk, color = "red",label=f"GD,bulk_scaling_{scaling_bulk}")

    for i in range(5):
        gd_sharpness = torch.load(f"{gd_directory}/eigs_{save_name_bulk}")[:,i]
        axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="red")
        if i== 4:
            axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="red",label=f"GD_{mode_bulk}_{scaling_bulk}_top5")





axs[0].legend()

#axs[1].set_xlim(0,2500)


axs[1].set_title("sharpness")
axs[1].legend()
axs[1].set_xlabel("iteration")



makedirs(f"{gd_directory}/figures", exist_ok=True)
#plt.show()
plt.savefig(f"{gd_directory}/figures/beta_GD_GDM_cifar10.png", bbox_inches='tight', pad_inches=0)

