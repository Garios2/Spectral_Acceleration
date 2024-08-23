import torch
import matplotlib.pyplot as plt
from os import environ,makedirs
import numpy as np

dataset = "cifar10"
arch = "fc-tanh"
loss = "mse"
gd_lr = 0.04
gd_eig_freq = 10
cmap = plt.get_cmap('viridis')
scaling = 1.0
momentum = "polyak"
middlePoint=40
gdm_lr = gd_lr/10
compare_flat_scaling = 0
compare_gd = 0
compare_gdm=0
compare_bulk_gd = 0
compare_gdm_beta_99 =0
compare_gdm_beta_50 =0
compare_gdm_warmup_beta = 0
compare_gdm_acc_09 = 0


sgd = 1

#gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"
fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=100, sharex=True)
gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"
gdm_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{gdm_lr}_beta_0.9"


if sgd == 1:
    dataset = "cifar10"
    color = "green"
    lr = 2.0
    BS = 256
    loss = 'ce'
    arch = 'resnet18'
    epoch = 200
    gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{lr}/BS_{BS}/epoch_{epoch}"
    acc_goal = 0.99
    if acc_goal == 0.99:
        gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{lr}/BS_{BS}/acc_{acc_goal}/epoch_{epoch}"
    mode='global_scaling'
    scaling=1.0
    nfilter=40
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss = torch.load(f"{gd_directory}/train_loss_{save_name}")
    axs[0].plot(gd_train_loss,label=f"SGD_lr = {lr},BS={BS}", color=color)
    axs[0].set_title(f"train loss_{dataset}_{loss}")

    for i in range(5):
        gd_sharpness = torch.load(f"{gd_directory}/eigs_{save_name}")[:,i]
        axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=color)
        if i== 4:
            axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=color)
    axs[1].axhline(2 / lr, linestyle='dotted',label="MSS")
    #axs[1].set_ylim(0,200)
    """
    color = "blue"
    lr = 0.1
    BS = 1000
    arch = 'cnn-relu'
    gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{lr}/BS_{BS}/acc_0.99/epoch_500"
    mode='global_scaling'
    scaling=1.0
    nfilter=20
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss = torch.load(f"{gd_directory}/train_loss_{save_name}")
    axs[0].plot(gd_train_loss,label=f"SGD_lr = {lr},BS={BS}", color=color)
    
    for i in range(5):
        gd_sharpness = torch.load(f"{gd_directory}/eigs_{save_name}")[:,i]
        axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=color)
        if i== 4:
            axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=color)
    axs[1].axhline(2 / lr, linestyle='dotted',label="MSS")    
    """


fig_name = f"_sGd-cifar10.png"

if compare_flat_scaling==1:
    dataset = "cifar10"
    color = "green"
    lr = 0.04
    BS = 1000
    loss = 'ce'
    arch = 'resnet18'
    mode_flat='global_scaling'
    gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{lr}/BS_{BS}/epoch_160"

    scaling_flat=10.0
    nfilter_flat=40
    save_name_flat = "{}_{}_top_{}_step".format(mode_flat, scaling_flat,nfilter_flat)
    gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_{save_name_flat}")
    train_loss_flat = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_flat))
    axs[0].plot(train_loss_flat, color = "blue",label=f"SGD,{mode_flat}_{scaling_flat}")

    for i in range(5):
        gd_sharpness = torch.load(f"{gd_directory}/eigs_{save_name_flat}")[:,i]
        axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="blue")
        if i== 4:
            axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="blue",label=f"SGD_{mode_flat}_{scaling_flat}_top5")


    mode_flat='flat_scaling_v1'

    save_name_flat = "{}_{}_top_{}_step".format(mode_flat, scaling_flat,nfilter_flat)
    gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_{save_name_flat}")
    train_loss_flat = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_flat))
    axs[0].plot(train_loss_flat, color = "purple",label=f"SGD,{mode_flat}_{scaling_flat}")

    for i in range(5):
        gd_sharpness = torch.load(f"{gd_directory}/eigs_{save_name_flat}")[:,i]
        axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="purple")
        if i== 4:
            axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="purple",label=f"SGD_{mode_flat}_{scaling_flat}_top5")

    mode_flat='flat_scaling_v2'

    save_name_flat = "{}_{}_top_{}_step".format(mode_flat, scaling_flat,nfilter_flat)
    gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_{save_name_flat}")
    train_loss_flat = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_flat))
    axs[0].plot(train_loss_flat, color = "#4682B4",label=f"SGD,{mode_flat}_{scaling_flat}")

    for i in range(5):
        gd_sharpness = torch.load(f"{gd_directory}/eigs_{save_name_flat}")[:,i]
        axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="#4682B4")
        if i== 4:
            axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color="#4682B4",label=f"GD_{mode_flat}_{scaling_flat}_top5")

    axs[1].axhline(2 / (scaling_flat*lr), linestyle='dotted',label="scaling_MSS")




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
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gdm_train_loss = torch.load(f"{gdm_directory}/train_loss_{save_name}")
    axs[0].plot(gdm_train_loss,label=f"GDM,beta=0.9", color="orange")

    for i in range(5):
        gdm_sharpness = torch.load(f"{gdm_directory}/eigs_{save_name}")[:,i]
        axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color="orange")
        if i== 4:
            axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color="orange",label="beta0.9")
    axs[1].axhline(2*(1.9) / gdm_lr, linestyle='dotted',label="MSS_GDM")

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
    #axs[1].axhline(2*(19.9) / gdm_lr, linestyle='dotted',label="MSS_GDM")

if compare_gdm_beta_50==1:
    color = "grey"
    lr = gd_lr/2
    gdm_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{lr}_beta_0.5"
    mode='global_scaling'
    scaling=1.0
    nfilter=20
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gdm_train_loss = torch.load(f"{gdm_directory}/train_loss_{save_name}")
    axs[0].plot(gdm_train_loss,label=f"GDM,beta=0.5", color=color,alpha = 0.5)

    for i in range(5):
        gdm_sharpness = torch.load(f"{gdm_directory}/eigs_{save_name}")[:,i]
        axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color=color)
        if i== 4:
            axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color=color,label="beta0.5")
    axs[1].axhline(2*(1.5) / lr, linestyle='dotted',label="MSS_GDM")

if compare_gdm_warmup_beta==1:
    color = "purple"
    lr = gd_lr/10
    gdm_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{lr}_beta_0.9/warmup_beta_0.99"
    mode='global_scaling'
    scaling=1.0
    nfilter=1
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)

    gdm_train_loss = torch.load(f"{gdm_directory}/train_loss_{save_name}")
    axs[0].plot(gdm_train_loss,label=f"GDM,warmup beta", color=color)

    for i in range(5):
        gdm_sharpness = torch.load(f"{gdm_directory}/eigs_{save_name}")[:,i]
        axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color=color)
        if i== 4:
            axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color=color,label="warmup_beta")
    axs[1].axhline(2*(1.9) / lr, linestyle='dotted',label="MSS_GDM")

if compare_gdm_acc_09 == 1:
    mode='global_scaling'
    scaling=1.0
    nfilter=20
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gdm_train_loss_1 = torch.load(f"{gdm_directory}/train_loss_{save_name}")
    gdm_train_loss_2 = torch.load(f"{gdm_directory}/acc_0.99/train_loss_{save_name}")
    gdm_train_loss = torch.cat((gdm_train_loss_1,gdm_train_loss_2))
    axs[0].plot(gdm_train_loss,label=f"GDM,beta=0.9", color="orange")

    for i in range(5):
        gdm_sharpness_1 = torch.load(f"{gdm_directory}/eigs_{save_name}")[:,i]
        gdm_sharpness_2 = torch.load(f"{gdm_directory}/acc_0.99/eigs_{save_name}")[:,i]
        gdm_sharpness = torch.cat((gdm_sharpness_1,gdm_sharpness_2))
        axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color="orange")
        if i== 4:
            axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color="orange",label="beta0.9")
    axs[1].axhline(2*(1.9) / gdm_lr, linestyle='dotted',label="MSS_GDM")

    color = "purple"
    lr = gd_lr/10
    gdm_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{lr}_beta_0.9/warmup_beta_0.99/acc_0.99"
    mode='global_scaling'
    scaling=1.0
    nfilter=1
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)

    gdm_train_loss = torch.load(f"{gdm_directory}/train_loss_{save_name}")
    axs[0].plot(gdm_train_loss,label=f"GDM,warmup beta", color=color)

    for i in range(5):
        gdm_sharpness = torch.load(f"{gdm_directory}/eigs_{save_name}")[:,i]
        axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color=color)
        if i== 4:
            axs[1].plot(torch.arange(len(gdm_sharpness)) * gd_eig_freq, gdm_sharpness, color=color,label="warmup_beta")
    axs[1].axhline(2*(1.9) / lr, linestyle='dotted',label="MSS_GDM")


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


axs[1].set_title("top eigenvalues")
axs[1].legend()
axs[1].set_xlabel("epochs")



makedirs(f"{gd_directory}/figures", exist_ok=True)
#plt.show()
plt.savefig(f"{gd_directory}/figures/{fig_name}", bbox_inches='tight', pad_inches=0)

