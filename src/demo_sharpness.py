import torch
import matplotlib.pyplot as plt
from os import environ,makedirs
import numpy as np

dataset = "cifar10"
arch = "fc-tanh"
loss = "mse"
gd_lr = 0.04
gd_eig_freq = 1
cmap = plt.get_cmap('viridis')
scaling = 1.0
momentum = "polyak"
middlePoint=1000
gdm_lr = gd_lr/10
compare_flat_scaling = 0
compare_gd =1
compare_gdm=0
compare_bulk_gd = 0
compare_gdm_beta_99 =0
compare_gdm_beta_50 =0
compare_gdm_warmup_beta = 0
compare_gdm_acc_09 = 0

compare_bulk_gd_1=0
gdm = 0

#gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"
fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=100, sharex=True)
gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"
gdm_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{gdm_lr}_beta_0.9"


if gdm == 1:
    dataset = "cifar10-5k"
    color = "green"
    lr = 0.004
    loss = 'ce'
    arch = 'cnn-relu'
    gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{lr}_beta_0.9/acc_0.99/bulk-only/warmupbeta_0.95"
    mode='global_scaling'
    scaling=1.0
    nfilter=20
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss = torch.load(f"{gd_directory}/train_loss_{save_name}")
    axs[0].plot(gd_train_loss,label=f"GDM_lr = {lr}", color=color)
    axs[0].set_title(f"train loss_{dataset}_{loss}")

    for i in range(10):
        gd_sharpness = torch.load(f"{gd_directory}/eigs_{save_name}")[:,i]
        axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=color)
        if i== 4:
            axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=color)
    axs[1].axhline(3.8 / (lr*scaling), linestyle='dotted',label="MSS_GDM")

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



if compare_bulk_gd_1 ==1:
    for nfilter in [40,100]:
        dataset = "cifar10"
        #color = "blue"
        color = cmap(nfilter/100)
        lr = 0.04
        loss = 'ce'
        arch = 'cnn-relu'
        gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{lr}/acc_0.99/bulk-only"
        mode='flat_scaling_v1'
        scaling=1.0
        #nfilter=20
        middlePoint=0

        save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
        gd_train_loss = torch.load(f"{gd_directory}/train_loss_{save_name}")
        axs[0].plot(torch.cat((torch.tensor([np.nan]*middlePoint), gd_train_loss)),label=f"Bulk-GD top{nfilter} lr = {lr}", color=color)
        axs[0].set_title(f"train loss_{dataset}_{loss}")

        for i in range(10):
            gd_sharpness = torch.load(f"{gd_directory}/eigs_{save_name}")[:,i]
            axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=color)
            if i== 4:
                axs[1].plot(middlePoint+torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=color)
        axs[1].axhline(2 / (lr*scaling), linestyle='dotted',label="MSS_GD")



if compare_gdm_beta_99==1:
    dataset = "cifar10-5k"
    color = "red"
    lr = 0.05
    beta=0.99
    loss = 'ce'
    arch = 'cnn-relu'
    gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/{momentum}/lr_{lr}_beta_{beta}/acc_0.99/bulk-only"
    mode='global_scaling'
    scaling=1.0
    nfilter=20
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss = torch.load(f"{gd_directory}/train_loss_{save_name}")
    axs[0].plot(gd_train_loss,label=f"GDM_lr = {lr}", color=color)
    axs[0].set_title(f"train loss_{dataset}_{loss}")

    for i in range(10):
        gd_sharpness = torch.load(f"{gd_directory}/eigs_{save_name}")[:,i]
        axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=color)
        if i== 4:
            axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=color)
    axs[1].axhline(3.99 / (lr*scaling), linestyle='dotted',label="MSS_GDM")

if compare_gd ==1:
    dataset = "cifar10"
    color = "red"
    lr = 0.05
    loss = 'mse'
    arch = 'cnn-relu'
    gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{lr}/acc_0.99/bulk-only"
    mode='global_scaling'
    scaling=1.0
    nfilter=10
    cubic = 0
    save_name = "{}_{}_top_{}_step".format(mode, scaling,nfilter)
    gd_train_loss = torch.load(f"{gd_directory}/train_loss_{save_name}")

    quadratic_update = torch.load(f"{gd_directory}/quadratic_update_{mode}_{scaling}_top_{nfilter}_step")
    bulk_update = torch.load(f"{gd_directory}/bulk_update_{mode}_{scaling}_top_{nfilter}_step")
    dom_update = torch.load(f"{gd_directory}/dom_update_{mode}_{scaling}_top_{nfilter}_step")
    bulk_grads = torch.load(f"{gd_directory}/filtered_grads_{mode}_{scaling}_top_{nfilter}_step")
    grads = torch.load(f"{gd_directory}/grads_{mode}_{scaling}_top_{nfilter}_step")
    dom_grads = grads.clone()-bulk_grads.clone()

    #fake_sharpness_1 = torch.load(f"{gd_directory}/fake_sharpness_one_{mode}_{scaling}_top_{nfilter}_step")
    #fake_sharpness_2 = torch.load(f"{gd_directory}/fake_sharpness_two_{mode}_{scaling}_top_{nfilter}_step")
    if cubic == 1:
        cubic_term = torch.load(f"{gd_directory}/cubic_term_{mode}_{scaling}_top_{nfilter}_step")
        dom_cubic = torch.load(f"{gd_directory}/dom_cubic_{mode}_{scaling}_top_{nfilter}_step")
        bulk_cubic = torch.load(f"{gd_directory}/bulk_cubic_{mode}_{scaling}_top_{nfilter}_step")
        print(dom_cubic.shape)
        print(bulk_cubic.shape)

        min_length = min(len(quadratic_update), len(cubic_term))

        # 截取较短长度的张量
        quadratic_update_short = quadratic_update[:min_length]
        cubic_term_short = cubic_term[:min_length]
        cubic_update = quadratic_update_short + cubic_term_short

        dom_update_short = dom_update[:min_length]
        print(dom_update_short.shape)
        dom_cubic_short = dom_cubic[:,0]

        dom_cubic_update = dom_update_short + dom_cubic_short

    b_update = torch.zeros_like(bulk_update)
    g_update =[]
    d_update=  torch.zeros_like(bulk_update)

    eigvals = torch.load(f"{gd_directory}/eigs_{mode}_{scaling}_top_{nfilter}_step")
    domnorm = [lr*torch.norm(u)**2 for u in dom_grads]# domnorm 在单个特征向量的时候就是c_i ** 2
    gnorm = [lr*torch.norm(u)**2 for u in grads]
    linear_update=  torch.zeros_like(bulk_update)
    linear_update[0] = gd_train_loss[0]
    # 对于每个元素，计算新的值
    for i in range(1,len(b_update)):
        b_update[i] = bulk_update[i] - bulk_update[i-1]
        d_update[i] = dom_update[i] - dom_update[i-1]
        linear_update[i] = linear_update[i-1] -gnorm[i-1]
    b_update[0] = 0
    d_update[0] = 0

    d_update = torch.cat((d_update[1:], torch.tensor([0])))

    #axs[0].plot(linear_update,label="linear",color = "yellow")

    domsecorder = [x+y for x,y in zip(d_update,domnorm)]
    #fake_sharpness = [y/(0.5*lr*x) for x,y in zip(domnorm,domsecorder)] # fake sharpness 很准
    bulknorm = [lr*torch.norm(u)**2 for u in bulk_grads]
    
    propo=0
    # 展示Dom和Bulk在梯度中的比例
    if propo==1:
        dom_propotion = [x/y for x,y in zip(domnorm,gnorm)] 
        bulk_propotion = [x/y for x,y in zip(bulknorm,gnorm)]
        axs[0].plot(dom_propotion,label=f"Dom_propo", color="green")
        axs[0].plot(bulk_propotion,label=f"bulk_propo", color="red")
    fig_name = f"GDM_BulkGD_-cifar10_{propo}.png"

    #print(d_update)
    axs[0].plot(gd_train_loss,label=f"GD lr = {lr}", color=color,alpha=0.3)
    axs[0].plot(quadratic_update,label=f"Quadratic approximation", color="orange",alpha=0.5)
    #axs[0].plot(dom_cubic_update,label=f"Dom cubic update", color="black",alpha=0.5)
    # 一二阶的更新
    axs[0].plot(dom_update,label=f"Dom update", color="black",alpha=0.5)
    axs[0].plot(bulk_update,label=f"Bulk update", color="blue",alpha=0.5)

    #axs[0].plot(cubic_term,color = "black",alpha = 1,label = "cubic term")

    #axs[0].plot(cubic_update,color = "green",alpha = 0.5,label = "Cubic approximation")
    #axs[0].set_ylim(0,3)
    #axs[0].set_xlim(-20,300)
    #axs[0].plot(dom_update,label=f"Dom_update", color="black")
    #axs[0].plot(bulk_update,label=f"Bulk_update", color="green")

    #axs[0].plot(domsecorder,label=f"second_order", color="purple")
    #axs[1].plot(fake_sharpness_1,color = "blue",alpha = 0.4)
    #axs[1].plot(fake_sharpness_2*2*lr,color = "green",alpha = 0.4)
    #axs[1].plot(fake_sharpness,color = "orange",alpha = 0.8)


    print(f"secondorder_sum: {torch.sum(torch.tensor(domsecorder))}")
    #print(f"cubic_sum: {torch.sum(torch.tensor(cubic_term))}")
   
    print(f"dom_update:{torch.sum(d_update[50:150])}")
    print(f"bulk_update:{torch.sum(b_update)}")
    print(f"quadratic_update:{torch.sum(b_update)+torch.sum(d_update) }")

    print(f"real_update:{gd_train_loss[-1] - gd_train_loss[0]}")
    axs[0].set_title(f"train loss_{dataset}_{loss}")

    for i in range(5):
        gd_sharpness = torch.load(f"{gd_directory}/eigs_{save_name}")[:,i]
        axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=color,alpha = 0.8)
        if i== 4:
            axs[1].plot(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, color=color)
    axs[1].axhline(2 / (lr*scaling), linestyle='dotted',label="MSS_GD",color=color)

    #axs[1].set_ylim(0,400)


if compare_flat_scaling==1:
    dataset = "cifar10-5k"
    lr = 0.004
    BS = 1000
    loss = 'ce'
    arch = 'cnn-relu'
    mode_flat='global_scaling'
    scaling_flat=2.0
    nfilter_flat=20
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
    scaling_flat=2.0
    nfilter_flat=20
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
    scaling_flat=2.0
    nfilter_flat=20
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

