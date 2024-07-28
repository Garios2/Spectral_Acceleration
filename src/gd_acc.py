from os import makedirs

import torch
from torch.nn.utils import parameters_to_vector

import argparse
from tqdm import tqdm
from archs import load_architecture
from utilities import get_gd_optimizer, get_gd_directory, get_loss_and_acc, compute_losses, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset,compute_gradient,\
        save_files_at_nstep,compute_flat_matrix
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
    def __init__(self, params, lr, mode=None, scaler=2.5):
        defaults = dict(lr=lr, mode=mode, scaler=scaler)
        super(AcD, self).__init__(params, defaults)
    
    def step(self, flat_matrix=None):
        full_grad = None

        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if full_grad is None:
                    full_grad = grad.view(-1)
                else:
                    full_grad = torch.cat([full_grad, grad.view(-1)])
        grad_flat = flat_matrix(full_grad) * group['scaler']


        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                #print("shape of grad:"+str(grad.shape))
                len_now = len(grad.view(-1))
                # 根据模式修改梯度
                if group['mode'] == 'global_scaling':
                    grad *= group['scaler']
                elif group['mode'] == 'flat_scaling' and flat_matrix is not None:
                    grad_tmp = grad_flat[:len_now]
                    grad_flat = grad_flat[len_now:]
                    grad_tmp = grad_tmp.view_as(grad)
                    grad = grad_tmp

                param.data.add_(-group['lr'], grad)





def main(dataset: str, arch_id: str, loss: str, opt: str, lr: float, max_steps: int, neigs: int = 0,
         physical_batch_size: int = 5000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = True, beta: float = 0.0, nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 5000, seed: int = 0):
    directory = get_gd_directory(dataset, lr, arch_id, seed, opt, loss, beta)
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)
    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).to(device)
    pretrained_dict = torch.load(f"{directory}/snapshot_6k_step")
    network.load_state_dict(pretrained_dict)



    len_of_param = len(parameters_to_vector((network.parameters())))

    torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))

    #optimizer = get_gd_optimizer(network.parameters(), opt, lr, beta)
    optimizer = AcD(params=network.parameters(), lr=lr, mode='flat_scaling',scaler=2.5)

    train_loss, test_loss, train_acc, test_acc = \
        torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    nfilter=10
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    eigvecs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len_of_param, neigs)
    gradients = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len_of_param)
   # param_flow = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len_of_param)
    #flat_matrix = torch.eye(len_of_param)
    trange = tqdm(range(max_steps))
    for step in trange: 
        train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
                                                           physical_batch_size)
        test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)

        

        if eig_freq != -1 and step % eig_freq == 0:
            eigs[step // eig_freq, :], eigvecs[step // eig_freq, :,:] = get_hessian_eigenvalues(network, loss_fn, train_dataset, neigs=neigs,
                                                                physical_batch_size=physical_batch_size)
            
            print("eigenvalues: ", eigs[step//eig_freq, :])
            #param_flow[step // eig_freq,:] = parameters_to_vector(network.parameters())
            gradients[step // eig_freq,:] = compute_gradient(network, loss_fn,train_dataset)
            flat_matrix = compute_flat_matrix(nfilter=nfilter,eigvecs=eigvecs[step // eig_freq, :,:])
            
        if iterate_freq != -1 and step % iterate_freq == 0:
            iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())

        if save_freq != -1 and step % save_freq == 0:
            save_files(directory, [("eigs", eigs[:step // eig_freq]), ("iterates", iterates[:step // iterate_freq]),
                                    ("grads", gradients[:step // eig_freq]),("eigvecs",eigvecs[:step // eig_freq]),
                                   ("train_loss", train_loss[:step]), ("test_loss", test_loss[:step]),
                                   ("train_acc", train_acc[:step]), ("test_acc", test_acc[:step])])

        print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")

        if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
            break

        optimizer.zero_grad()
        for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
            loss = loss_fn(network(X.to(device)), y.to(device)) / len(train_dataset)
            loss.backward()
        optimizer.step(flat_matrix=flat_matrix)

    save_files_at_nstep(directory,
                     [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                     ("grads", gradients[:(step + 1) // eig_freq]),("eigvecs",eigvecs[:(step + 1) // eig_freq]),
                      ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                      ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1])], step="6k-10k-flat-scaling-2.5")
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_6k-10k-flat-scaling-2.5")


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