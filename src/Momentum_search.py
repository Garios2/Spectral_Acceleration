from os import makedirs
import matplotlib.pyplot as plt
import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
import argparse

from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from archs import load_architecture
from utilities import get_gd_optimizer, get_gd_directory, get_loss_and_acc, compute_losses, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset,compute_gradient,\
        save_files_at_nstep,compute_flat_matrix,weights_init
from data import load_dataset, take_first, DATASETS
import os
"""
export DATASETS=/data/users/zhouwenjie/workspace/Spectral_Acceleration/data
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RESULTS=/data/users/zhouwenjie/workspace/Spectral_Acceleration/results
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class AcD(torch.optim.Optimizer):
    def __init__(self, params, lr, mode=None, scaler=1):
        defaults = dict(lr=lr, mode=mode, scaler=scaler)
        super(AcD, self).__init__(params, defaults)

    def step(self, flat_matrix=None):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data

                # 根据模式修改梯度
                if group['mode'] == 'global_scaling':
                    grad *= group['scaler']
                elif group['mode'] == 'flat_scaling' and flat_matrix is not None:
                    grad = flat_matrix(grad) * group['scaler']

                param.data.add_(-group['lr'], grad)





def main(dataset: str, arch_id: str, loss: str, opt: str, lr: float, max_steps: int, neigs: int = 0,
         physical_batch_size: int = 5000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = True, beta: float = 0.0, nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 5000, seed: int = 0):
    results_dir = os.environ["RESULTS"]
    directory = f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/{opt}/grad_search"
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)
    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)
    train_dataset = [(x.to(device), y.to(device)) for x, y in train_dataset]
    test_dataset = [(x.to(device), y.to(device)) for x, y in test_dataset]
    

    loss_fn, acc_fn = get_loss_and_acc(loss)

    
    #network = load_architecture(arch_id, dataset).to(device)
    #pretrained_dict = torch.load(f"{directory}/snapshot_6k_step")
    #network.load_state_dict(pretrained_dict)

    network = load_architecture(arch_id, dataset).to(device)

    #len_of_param = len(parameters_to_vector((network.parameters())))

    torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))
    fig = plt.figure(figsize=(10, 10), dpi=100)
    gs = GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    cmap = plt.get_cmap('viridis')

    for seed in [1,2,3,4,5]:
        torch.manual_seed(seed)
        print(f"{lr}-current lr")
        network = load_architecture(arch_id, dataset).to(device)
        #network.apply(weights_init(gain=))
        optimizer = get_gd_optimizer(network.parameters(), opt, lr, beta)
        #optimizer = AcD(params=network.parameters(), lr=lr, mode='global_scaling',scaler=1)
        eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
        train_loss, test_loss, train_acc, test_acc = \
            torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
        iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))

        # param_flow = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len_of_param)
        #flat_matrix = torch.eye(len_of_param)
        for step in range(max_steps): 
            train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
                                                            physical_batch_size)
            test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)

            print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")

            if eig_freq != -1 and step % eig_freq == 0:
                eigs[step // eig_freq, :], _ = get_hessian_eigenvalues(network, loss_fn, train_dataset, neigs=neigs,
                                                                    physical_batch_size=physical_batch_size)
                print("eigenvalues: ", eigs[step//eig_freq, :])
                #param_flow[step // eig_freq,:] = parameters_to_vector(network.parameters())
                
                


            if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
                break

            
            optimizer.zero_grad()
            for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
                loss = loss_fn(network(X.to(device)), y.to(device)) / len(train_dataset)
                loss.backward()
            optimizer.step()
        
        #ax1.plot(train_loss,color = cmap(i/25))
        name = "seed_is_{}".format(seed)
        save_files_at_nstep(directory,
                        [ ("eigs", eigs[:(step + 1) // eig_freq]),("iterates", iterates[:(step + 1) // iterate_freq]),
                        ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),], step=name)
        ax1.plot(torch.arange(1000),train_loss[:1000], color = cmap(seed/5))
        sharpness = eigs[:,0]
        #ax2.scatter(torch.arange(len(sharpness))*eig_freq, sharpness, 'o-', color=cmap(seed/5))
        ax2.plot(torch.arange(1000),sharpness[:1000], 'o-', color=cmap(seed/5))
    
    ax2.set_ylabel("sharpness")
    ax2.axvline(2./lr,linestyle='dotted')
    ax1.set_ylabel("train loss")
    ax2.set_xlabel("iteration")
    dir = f"{directory}/figures"
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig_save = f"{directory}/figures/Momentun_search.svg"
    plt.savefig(fig_save, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("lr", type=float, help="the learning rate")
    parser.add_argument("max_steps", type=int, help="the maximum number of gradient steps to train for")
    parser.add_argument("--opt", type=str, choices=["gd", "polyak", "nesterov"],
                        help="which optimization algorithm to use", default="gd")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--beta", type=float, help="momentum parameter (used if opt = polyak or nesterov)")
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=2000)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value")
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value")
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--nproj", type=int, default=0, help="the dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="the frequency at which we save random projections of the iterates")
    parser.add_argument("--abridged_size", type=int, default=5000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save resuls")
    parser.add_argument("--save_model", type=bool, default=True,
                        help="if 'true', save model weights at end of training")
    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, max_steps=args.max_steps,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq,
         iterate_freq=args.iterate_freq, save_freq=args.save_freq, save_model=args.save_model, beta=args.beta,
         nproj=args.nproj, loss_goal=args.loss_goal, acc_goal=args.acc_goal, abridged_size=args.abridged_size,
         seed=args.seed)