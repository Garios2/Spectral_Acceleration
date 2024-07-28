import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# import torch.linalg
from utils import FakeDL
import algos.pyhessian as pyhessian
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from utils import kp_2d
import matplotlib as mpl
from tqdm import tqdm
from matplotlib import rcParams
import os
from matplotlib.gridspec import GridSpec
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import pickle
# mpl.use('Agg')
config = {
    "font.family":'Times New Roman',
    "axes.unicode_minus": False
}
rcParams.update(config)
#plt.rc('text', usetex=True)
#params= {'text.latex.preamble' : [r'\usepackage{amsmath}',r'\usepackage{amssymb}', r'\usepackage{bm}']}
#plt.rcParams.update(params)
plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath} \usepackage{amssymb}"

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dev)


def plot_sharpness_traj(ss, eos, fig_size, exp_name):
    plt.figure(figsize=fig_size)
    plt.axhline(eos, label=r"EoS Threshold ($2/\eta$)", linestyle='--', color='black')
    plt.scatter(np.arange(len(ss)), ss, label="Numerical Sharpness", s=7, zorder=100)
    plt.legend(loc=1)
    plt.xlabel('Iterations')
    plt.ylabel('Sharpness')
    plt.tight_layout()
    plt.savefig(f"./figs/{exp_name}_sharpness.pdf")
    plt.show()


def plot_loss_traj(losses, fig_size, exp_name, logy=False):
    plt.figure(figsize=fig_size)
    plt.plot(np.arange(len(losses)), losses, label="Loss", zorder=100)
    plt.legend(loc=1)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    if logy:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f"./figs/{exp_name}_loss.pdf")
    plt.show()



class LinearNet(nn.Module):
    def __init__(self, X_size,L, d):
        super(LinearNet, self).__init__()
        self.L = L
        self.X_size = X_size
        self.layers = []
        for i in range(L):
            if i==0:
                self.layers.append(nn.Linear(X_size, d, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])

            elif i == L-1:
                self.layers.append(nn.Linear(d, X_size, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])
            else:
                self.layers.append(nn.Linear(d, d, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])

    def forward(self, x):
        for i in range(self.L):
            x = self.layers[i](x)
        return x


class LinearReLUNet(nn.Module):
    def __init__(self, X_size,L, d):
        super(LinearReLUNet, self).__init__()
        self.L = L
        self.X_size = X_size
        self.layers = []
        for i in range(L):
            if i == 0:
                self.layers.append(nn.Linear(X_size, d, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])

            elif i == L - 1:
                self.layers.append(nn.Linear(d, X_size, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])
            else:
                self.layers.append(nn.Linear(d, d, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])

    def forward(self, x):
        for i in range(self.L):
            x = self.layers[i](x)
            if i != self.L - 1:
                x = nn.functional.relu(x)
        return x


class LinearELUNet(nn.Module):
    def __init__(self, X_size,L, d):
        super(LinearELUNet, self).__init__()
        self.L = L
        self.X_size = X_size
        self.layers = []
        for i in range(L):
            if i == 0:
                self.layers.append(nn.Linear(X_size, d, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])

            elif i == L - 1:
                self.layers.append(nn.Linear(d, X_size, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])
            else:
                self.layers.append(nn.Linear(d, d, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])
    def forward(self, x):
        for i in range(self.L):
            x = self.layers[i](x)
            if i != self.L - 1:
                x = nn.functional.elu(x)
        return x


class LinearSigmoidNet(nn.Module):
    def __init__(self, X_size,L, d):
        super(LinearSigmoidNet, self).__init__()
        self.L = L
        self.X_size = X_size
        self.layers = []
        for i in range(L):
            if i == 0:
                self.layers.append(nn.Linear(X_size, d, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])

            elif i == L - 1:
                self.layers.append(nn.Linear(d, X_size, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])
            else:
                self.layers.append(nn.Linear(d, d, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])

    def forward(self, x):
        for i in range(self.L):
            x = self.layers[i](x)
            if i != self.L - 1:
                x = nn.functional.sigmoid(x)
        return x


class LinearTANHNet(nn.Module):
    def __init__(self, X_size,L, d):
        super(LinearTANHNet, self).__init__()
        self.L = L
        self.X_size = X_size
        self.layers = []
        for i in range(L):
            if i == 0:
                self.layers.append(nn.Linear(X_size, d, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])

            elif i == L - 1:
                self.layers.append(nn.Linear(d, X_size, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])
            else:
                self.layers.append(nn.Linear(d, d, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])

    def forward(self, x):
        for i in range(self.L):
            x = self.layers[i](x)
            if i != self.L - 1:
                x = nn.functional.tanh(x)
        return x


class Linearleaky_ReLUNet(nn.Module):
    def __init__(self, X_size,L, d):
        super(Linearleaky_ReLUNet, self).__init__()
        self.L = L
        self.X_size = X_size
        self.layers = []
        for i in range(L):
            if i == 0:
                self.layers.append(nn.Linear(X_size, d, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])

            elif i == L - 1:
                self.layers.append(nn.Linear(d, X_size, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])
            else:
                self.layers.append(nn.Linear(d, d, bias=False))
                self.add_module("W{}".format(i + 1), self.layers[-1])

    def forward(self, x):
        for i in range(self.L):
            x = self.layers[i](x)
            if i != self.L - 1:
                x = nn.functional.leaky_relu(x)
        return x


class Spectral_acc(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Spectral_acc, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Spectral_acc, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['weight_decay'] != 0:
                    d_p.add_(group['weight_decay'], p.data)
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p)
                    if group['nesterov']:
                        d_p = d_p.add(group['momentum'], buf)
                    else:
                        d_p = buf

                # 在这里，你可以将梯度乘以你的矩阵A
                # 假设你的矩阵A是一个函数，它接受一个参数（当前的迭代次数或其他信息）
                # 并返回一个矩阵
                A = get_A(...)
                d_p = torch.matmul(A, d_p)

                p.data.add_(-group['lr'], d_p)

def compute_eigeninfo(net, dl, topn, criterion):
    # Computing the top n eigenvectors using Power Iteration Method.
    net.eval()
    hessian_comp = pyhessian.hessian(net, criterion, dataloader=dl, cuda=False if dev == "cpu" else True)
    eigenvals, eigenvecs = hessian_comp.eigenvalues(top_n=topn, tol=1e-4, maxIter=10000)
    dir_hessian = hessian_comp.dirctional_hessian()
    # print(eigenvals)
    net.train()
    return eigenvals,eigenvecs,dir_hessian


def compute_hvp(net, dl, topn, criterion, v):
    # Computing the top n eigenvectors using Power Iteration Method.
    net.eval()
    hessian_comp = pyhessian.hessian(net, criterion, dataloader=dl, cuda=False if dev == "cpu" else True)
    # print("conducting HVP with v{}".format(v.shape))
    val, hvp = hessian_comp.dataloader_hv_product(v)
    return val, hvp


def eigenthings_tensor_utils(t, device=None, out_device='cpu', symmetric=False, topn=-1):
    t = t.to(device)
    if topn >= 0:
        _, eigenvals, eigenvecs = torch.svd_lowrank(t, q=min(topn, t.size()[0], t.size()[1]))
        eigenvecs.transpose_(0, 1)
    else:
        if symmetric:
            eigenvals, eigenvecs = torch.symeig(t, eigenvectors=True)  # pylint: disable=no-member
            eigenvals = eigenvals.flip(0)
            eigenvecs = eigenvecs.transpose(0, 1).flip(0)
        else:
            _, eigenvals, eigenvecs = torch.svd(t, compute_uv=True)  # pylint: disable=no-member
            eigenvecs = eigenvecs.transpose(0, 1)
    return eigenvals, eigenvecs

def get_n_cifar(n):
    # 加载CIFAR-10数据集
    dataset = CIFAR10(root='/data/users/zhouwenjie/workspace/Spectral_Acceleration/data', train=True, download=True, transform=ToTensor())

    # 获取图像和标签
    images, labels = dataset.data, np.array(dataset.targets)

    # 指定你想要的类别
    categories = list(range(10))  # 假设你想要的类别为0到9

    # 初始化空列表来保存选定的图像和标签
    selected_images = []
    selected_labels = []

    # 对每个类别进行处理
    for category in categories:
        # 找出该类别的索引
        indices_category = np.where(labels == category)[0]

        # 从该类别中随机选取n/10张图片
        selected_indices_category = np.random.choice(indices_category, int(n/10), replace=False)

        # 添加选定的图像和标签到列表中
        selected_images.extend(images[selected_indices_category])
        selected_labels.extend(labels[selected_indices_category])

    # 数据现在已经被选出，n张图片，每个类别n/10张
    train_images_flat = np.array(selected_images).reshape(len(selected_images), -1).astype('float32')
    X = torch.tensor(train_images_flat, dtype=torch.float32) / 255.0  # 归一化图像数据到[0, 1]

    Y = torch.tensor(selected_labels,dtype=torch.int64)
    Y = Y.view(n,1)
    return X,Y


def quadratic_loss(Y_hat, Y):
    """Batched quadratic loss"""
    diff = torch.square(Y - Y_hat)
    loss = diff.sum() * 0.5 / len(Y)
    if len(Y_hat.shape) != 1:
        loss
    return loss

def square_root_loss(Y_hat, Y):
    diff = torch.sqrt(1+torch.square(Y - Y_hat))
    loss = diff.sum() / len(Y)
    if len(Y_hat.shape) != 1:
        loss
    return loss


def get_Wprod_singular_space(Ws):
    W_prod = Ws[0].clone()
    for W in Ws[1:]:
        W_prod = W.matmul(W_prod)

    u_prod, s_prod, v_prod = np.linalg.svd(W_prod)
    return W_prod, u_prod, s_prod, v_prod


'''
    for d in [75,80]:
        for L in [7,8]:
'''

# { 45, 50, 55, 60, 75, 80}
def plot_h(net_type, d, L,n,X_size,momen,sharp,iter, flag,warmup,rep,freq,X,Y,rescale):
    print("current acti={},layer={},width={}".format(net_type, L, d))
    seed, eval_start, eval_end = 0, 0, 4000
    #freq = 1
    # d = 10 # layer width
    #n = 5  # # of samples
    # L = 2 # # of layers
    res = {}
    #X_size = 2




    for cnt in range(iter):
        if momen==0:
            eta = 2 / sharp
        else:
            eta = 3.8 / sharp

        #torch.manual_seed(seed)
        hessian_topn = 4
        epochs = eval_end
        hessian_eval_epochs = list(range(eval_start, eval_end, freq))
        if net_type == "ReLU":
            net = LinearReLUNet(X_size, L, d)  #
        elif net_type == "Linear":
            net = LinearNet(X_size, L, d)  #
        elif net_type == "Sigmoid":
            net = LinearSigmoidNet(X_size, L, d)  #
        elif net_type == "tanh":
            net = LinearTANHNet(X_size, L, d)  #
        else:
            net = LinearELUNet(X_size, L, d)  #

        # 初始化权重
        for layer in net.layers:
            torch.nn.init.xavier_normal_(layer.weight, gain=rescale)
            print("initial weight goes:{}".format(layer.weight))

        # 使用DataParallel并将模型移动到GPU
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = net.to('cuda:0')  # 将模型移动到'cuda:0'
            #net = nn.DataParallel(net, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

        X = X.to(dev)  # 将数据移动到'cuda:0'
        Y = Y.to(dev)  # 将数据移动到'cuda:0'
        dataloader = FakeDL(X, Y, dev)
        loss_record = []
        eigenvals_record = []
        eigenvecs_record = []

        weight_record = []
        weight_prod_record = []
        Y_record = []
        gradient_record = []

        # criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(net.parameters(), lr=eta, momentum=momen, nesterov=False,dampening=0)
        if warmup==1:
            scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.0000000001, end_factor=1, total_iters=int(eval_end/2))

        trange = tqdm(range(epochs))
        for epoch in trange:  # loop over the dataset multiple times

            # learning rate drop

            optimizer.zero_grad()
            
            Y_hat = net(X)
            loss = quadratic_loss(Y_hat, Y) # 损失函数
            #print("Y_hat={},loss={}".format(Y_hat,loss))
            loss.backward()
            optimizer.step()
            #print(optimizer.param_groups[0]['lr'])

            loss_record.append(loss.item())
            if warmup==1:
                scheduler.step()
            if epoch in hessian_eval_epochs:
                gradients = []

                for param in net.parameters():
                    gradients.append(param.grad.clone())
                gradient_record.append(gradients)

                eigenvals,eigenvecs,dir_hessians = compute_eigeninfo(net, dataloader, hessian_topn, quadratic_loss)
                eigenvals_record.append(eigenvals)
                eigenvecs_record.append(eigenvecs)

                Ws = [W.weight.data.clone() for W in net.layers]
                W_prod = Ws[0].clone()
                for W in Ws[1:]:
                    W_prod = W.matmul(W_prod)

                weight_record.append(Ws)
                weight_prod_record.append(W_prod)
                Y_record.append(Y_hat)


            if epoch % 3 == 0:

                trange.set_description_str("Epoch {}  Loss: {:.6g} Sharpness: {:.6g}".format(epoch, loss.item(),
                                                                                             eigenvals_record[
                                                                                                 -1][
                                                                                                 0] if len(
                                                                                                 eigenvals_record) > 0 else -1))
                """
                                if epoch % 120 == 0:
                    eta = 0.656* eta
                    optimizer = torch.optim.SGD(net.parameters(), lr=eta, momentum=momen, nesterov=False)
                """

            sharpness_array = np.array([v[0] for v in eigenvals_record])




        res[cnt] = [loss_record,sharpness_array, hessian_eval_epochs,Y_record,eigenvals_record,eigenvecs_record,gradient_record]

    #res_relu_deep = copy.deepcopy(res)


    #f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 10), gridspec_kw={'height_ratios': [0.6, 1,1]})
    fig = plt.figure()
    gs = GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)


    colors = ['red', 'blue', 'purple']
    for i,cnt in enumerate(res):
        loss_record,sharpness_array, hessian_eval_epochs,Y_record ,eigenvals_record,eigenvecs_record,gradient_record = res[cnt]
        #print(len(directionalH_array))
        if momen==0:
            loss_gap = loss_record
            #loss_gap = [u - 1 for u in loss_record]
            ax1.plot(np.arange(len(loss_gap)), np.array(loss_gap),
                     label=r'$\eta=2/{}$'.format(sharp))


            ax3.scatter(hessian_eval_epochs, sharpness_array, s=5)
            if warmup==1:
                x1 = np.linspace(1,eval_end,eval_end)
                y1 = np.where(x1<eval_end/2,(eval_end/2)/x1 * sharp,sharp)
                ax3.plot(x1,y1,linestyle='dotted')
            else:
                ax3.axhline(sharp, linestyle='--', color='black')
            ax3.text(epochs * 0.95, sharp , r'$\lambda={}$'.format(sharp), ha='right')

            if rep == 1:
                p = [u.cpu().detach().numpy() - Y.cpu().detach().numpy() for u in Y_record]

                p_new = np.squeeze(np.array(p)).tolist()
                q = [2 / ((2/sharp) * v) for v in sharpness_array]
                axx.axvline(x=0, color='black')
                axx.axhline(y=1, color='black')
                axx.plot(loss_record, q, 'bo-', alpha=0.8, linewidth=1,color=colors[i])

        else:
            loss_gap = loss_record
            #loss_gap = [u - 1 for u in loss_record]
            ax1.plot(np.arange(len(loss_gap)), np.array(loss_gap))

            ax3.scatter(hessian_eval_epochs, sharpness_array, s=5)
            if warmup==1:
                x1 = np.linspace(1,eval_end,eval_end)
                y1 = np.where(x1<eval_end/2,(eval_end/2)/x1 * sharp,sharp)
                ax3.plot(x1,y1,linestyle='dotted')
            else:
                ax3.axhline(sharp, linestyle='--', color='black')

        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        ax1.legend(loc=1)


        ax3.legend(loc=4)
        ax3.set_ylabel('Sharpness')
        ax3.set_xlabel('iteration')
        # ax2.set_ylim(0, 43)
        title = 'd={}_w={}_CIFAR_2_acti={}_Momen={}_rescale={}'.format(L, d, net_type, momen, rescale)
        ax1.set_title(title)
        plt.tight_layout()
        plt.savefig(f"/data/users/zhouwenjie/workspace/Spectral_Acceleration/results/Toynn/-{title}.png", bbox_inches='tight', pad_inches=0)

        fig_new = plt.figure()
        ax_new = fig_new.add_subplot(1, 1, 1)
        n = len(eigenvals_record)
        cmap = plt.get_cmap('viridis')
        for i in range(4):
            # 提取第i维的数据
            data_i = [item[i] for item in eigenvals_record]
            
            # 计算颜色
            color = cmap(i / 3)  # 5是因为我们有6条线，所以我们需要在0到1之间均匀地选择6个值
            
            # 绘制线
            ax_new.plot(range(n), data_i, color=color)
        plt.savefig(f"/data/users/zhouwenjie/workspace/Spectral_Acceleration/results/Toynn/eigenvalues.png", bbox_inches='tight', pad_inches=0)

        fig_2= plt.figure()
        ax_2 = fig_2.add_subplot(1, 1, 1)
        n = len(eigenvals_record)
        for i in range(4):
            # 提取第i维的数据
            data_i = [torch.dot(torch.tensor(eigenvecs_record[j][0]).flatten(), gradients[j].flatten()).item() for j in range(n)]       
            color = cmap(i / 3)  # 3是因为我们有4条线，所以我们需要在0到1之间均匀地选择4个值
            
            # 绘制线
            ax.plot(range(n), data_i, color=color)
        plt.savefig(f"/data/users/zhouwenjie/workspace/Spectral_Acceleration/results/Toynn/eigenvec_grad_dot.png", bbox_inches='tight', pad_inches=0)









if __name__ == '__main__':

    #para
    n= 3000# 数据点个数
    X_size=3072
    warm = 0
    L=3
    d=200
    rescale=1
    net_type = "tanh"
    momentum = 0

    # data
    X, Y = get_n_cifar(n)

    #X, Y = X.to(dev), Y.to(dev)
    for net_type in ["tanh"]:

        plot_h(net_type=net_type, d=d, L=L, n=n, X_size=X_size, momen=momentum, sharp=10000,iter=1 ,flag=0,
                warmup=0,rep=0,freq=10,X=X,Y=Y,rescale=rescale)
        #plot_h(net_type=net_type, d=d, L=L, n=n, X_size=X_size, momen=0.9, sharp=4000,iter=2 ,flag=0,
                #warmup=0,rep=0,freq=4,X=X,Y=Y,rescale=1)

        #plot_h(net_type=net_type, d=d, L=L, n=n, X_size=X_size, momen=0, sharp=500,iter=3 ,flag=0,
                #warmup=0,rep=0,freq=4,X=X,Y=Y,rescale=0.4)
        #plot_h(net_type=net_type, d=d, L=L, n=n, X_size=X_size, momen=0.9, sharp=500,iter=3 ,flag=0,
                #warmup=0,rep=0,freq=4,X=X,Y=Y,rescale=0.4)

        #plot_h(net_type=net_type, d=d, L=L, n=n, X_size=X_size, momen=0, sharp=500,iter=3 ,flag=0,
                #warmup=0,rep=0,freq=4,X=X,Y=Y,rescale=0.6)
        #plot_h(net_type=net_type, d=d, L=L, n=n, X_size=X_size, momen=0.9, sharp=500,iter=3 ,flag=0,
                #warmup=0,rep=0,freq=4,X=X,Y=Y,rescale=0.6)

    #plt.show()

