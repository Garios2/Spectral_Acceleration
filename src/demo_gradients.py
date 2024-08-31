import torch
import matplotlib.pyplot as plt
from os import environ,makedirs
import numpy as np

dataset = "cifar10-5k"
arch = "cnn-relu"
loss = "ce"
gd_lr = 0.04
gd_eig_freq = 100
cmap = plt.get_cmap('viridis')
scaling = 1.0
momentum = "polyak"
middlePoint=0
gdm_lr = gd_lr/10
compare_flat_scaling = 0
compare_gdm_beta_99 =0
compare_gdm_beta_50 =0

compare_bulk_gd_1 = 1
compare_bulk_gd_3 = 0
compare_sgd =0
compare_gd =1

compare_gdm=1
compare_gdm_warmup_beta=0
#gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"
#fig, axs = plt.subplots(3, 1, figsize=(15, 15), dpi=100, sharex=True)
#axs[1], axs[2] = fig.subplots(2, 1, sharey=True)
fig = plt.figure(figsize=(15, 15), dpi=100)

ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
ax3 = fig.add_subplot(3, 1, 3, sharex=ax1, sharey=ax2)

axs = [ax1, ax2, ax3]


gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"
gdm_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{gdm_lr}_beta_0.9"




fig.suptitle("CIFAR10-5K & CNN-relu & CE loss & Bulk-GD & GDM & Top20 filtered ", y=0.95, fontsize=16)


axs[0].set_title("train loss")
#axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[2].set_yscale('log')


if compare_gd == 1:
    color = "green"
    mode='global_scaling'
    scaling=1.0

    lr =0.04
    
    nfilter=20
    gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{lr}/acc_0.99/bulk-only"

    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss = torch.load(f"{gd_directory}/train_loss_{save_name}")
    gradient = torch.load(f"{gd_directory}/grads_{save_name}")
    g_norm = [torch.norm(u) for u in gradient]
    axs[0].plot(gd_train_loss,label=f"GD,eta={gd_lr}", color=color)
    axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)

    filtered_gradient = torch.load(f"{gd_directory}/filtered_grads_{mode}_{scaling}_top_{nfilter}_step")
    g_norm = [torch.norm(u) for u in filtered_gradient]
    axs[2].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)


if compare_sgd == 1:
    color = "green"
    mode='global_scaling'
    scaling=1.0

    lr =0.04
    
    nfilter=20
    BS = 1000
    epoch = 3000
    gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}/BS_{BS}/epoch_{epoch}"
    
    gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{lr}/acc_0.99"

    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss = torch.load(f"{gd_directory}/train_loss_{save_name}")
    gradient = torch.load(f"{gd_directory}/grads_{save_name}")
    g_norm = [torch.norm(u) for u in gradient]
    axs[0].plot(gd_train_loss,label=f"GD,eta={gd_lr}", color=color)
    axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)

    filtered_gradient = torch.load(f"{gd_directory}/filtered_grads_{mode}_{scaling}_top_{nfilter}_step")
    g_norm = [torch.norm(u) for u in filtered_gradient]
    axs[2].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)

if compare_flat_scaling==1:
    color = "purple"
    mode='flat_scaling_v1'
    scaling=3.0
    nfilter=20
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_{save_name}")
    train_loss_flat = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_flat))
    gradient = torch.load(f"{gd_directory}/grads_{save_name}")
    g_norm = [torch.norm(u) for u in gradient]
    nan_list = [np.nan] * int(middlePoint/gd_eig_freq)
    g_norm = nan_list + g_norm
    axs[0].plot(train_loss_flat, color = color,label=f"SGD,{mode}_{scaling}")
    axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)
    filtered_gradient = torch.load(f"{gd_directory}/filtered_grads_{mode}_{scaling}_top_{nfilter}_step")
    g_norm = [torch.norm(u) for u in filtered_gradient]
    axs[2].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)

 
    color = "red"
    mode='flat_scaling_v2'
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_{save_name}")
    train_loss_flat = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_flat))
    gradient = torch.load(f"{gd_directory}/grads_{save_name}")
    g_norm = [torch.norm(u) for u in gradient]
    nan_list = [np.nan] * int(middlePoint/gd_eig_freq)
    g_norm = nan_list + g_norm
    axs[0].plot(train_loss_flat, color = color,label=f"SGD,{mode}_{scaling}")
    axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)
    filtered_gradient = torch.load(f"{gd_directory}/filtered_grads_{mode}_{scaling}_top_{nfilter}_step")
    g_norm = [torch.norm(u) for u in filtered_gradient]
    axs[2].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)

    color = "blue"
    mode='global_scaling'
    scaling=3.0
    nfilter=20
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_{save_name}")
    train_loss_flat = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_flat))
    gradient = torch.load(f"{gd_directory}/grads_{save_name}")
    g_norm = [torch.norm(u) for u in gradient]
    nan_list = [np.nan] * int(middlePoint/gd_eig_freq)
    g_norm = nan_list + g_norm
    axs[0].plot(train_loss_flat, color = color,label=f"SGD,{mode}_{scaling}")
    axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)
    filtered_gradient = torch.load(f"{gd_directory}/filtered_grads_{mode}_{scaling}_top_{nfilter}_step")
    g_norm = [torch.norm(u) for u in filtered_gradient]
    axs[2].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)



if compare_bulk_gd_1==1:
    for nfilter in [5,10,20,40,100]:
        color = cmap(nfilter/100)
        mode='flat_scaling_v1'
        scaling=1.0
        middlePoint=0
        lr=0.04
        gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{lr}/acc_0.99/bulk-only"

        save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
        gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_{save_name}")
        train_loss_flat = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_flat))
        gradient = torch.load(f"{gd_directory}/grads_{save_name}")
        g_norm = [torch.norm(u) for u in gradient]
        nan_list = [np.nan] * int(middlePoint/gd_eig_freq)
        g_norm = nan_list + g_norm
        axs[0].plot(train_loss_flat, color = color,label=f"Bulk-GD,top{nfilter} filtered,{lr*scaling}")
        axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)
        filtered_gradient = torch.load(f"{gd_directory}/filtered_grads_{mode}_{scaling}_top_{nfilter}_step")
        g_norm = [torch.norm(u) for u in filtered_gradient]
        g_norm = nan_list + g_norm

        axs[2].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)
            


if compare_gdm ==1:
    color = "red"
    mode='global_scaling'
    scaling=1.0
    nfilter=20
    lr=0.004
    middlePoint=0
    gd_directory=f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{lr}_beta_0.9/acc_0.99/bulk-only/warmupbeta_0.95"
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss_flat = torch.load(f"{gd_directory}/train_loss_{save_name}")
    train_loss_flat = torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss_flat))
    gradient = torch.load(f"{gd_directory}/grads_{save_name}")
    g_norm = [torch.norm(u) for u in gradient]
    nan_list = [np.nan] * int(middlePoint/gd_eig_freq)
    g_norm = nan_list + g_norm
    axs[0].plot(train_loss_flat, color = color,label=f"GDM,{lr}")
    axs[1].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)
    filtered_gradient = torch.load(f"{gd_directory}/filtered_grads_{mode}_{scaling}_top_{nfilter}_step")
    g_norm = [torch.norm(u) for u in filtered_gradient]
    g_norm = nan_list + g_norm

    axs[2].plot(torch.arange(len(g_norm)) * gd_eig_freq, g_norm, color=color)


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






#axs[1].set_xlim(0,2500)
axs[0].legend()
axs[0].set_xlabel("iteration")

axs[1].set_title("gradient norm")
axs[1].legend()
axs[1].set_xlabel("iteration")
axs[2].set_title("filtered gradient norm")
axs[1].legend()
axs[1].set_xlabel("iteration")


makedirs(f"{gd_directory}/figures", exist_ok=True)
#plt.show()
plt.savefig(f"{gd_directory}/figures/gradientnorm_bulk_GDM_GD_cifar10_6.png", bbox_inches='tight', pad_inches=0)

