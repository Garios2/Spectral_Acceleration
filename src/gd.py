from os import makedirs

import torch
from torch.nn.utils import parameters_to_vector

import argparse
from archs import load_architecture
from torch.optim import SGD
from utilities import get_gd_optimizer, get_gd_directory, get_loss_and_acc, compute_losses, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset,compute_gradient,\
        save_files_at_nstep,compute_flat_matrix,get_directional_Hessian,compute_hvp,lanczos,compute_third_order_hvp
from data import load_dataset, take_first, DATASETS
import os
"""
export DATASETS=/data/users/zhouwenjie/workspace/Spectral_Acceleration/data
export RESULTS=/data/users/zhouwenjie/workspace/Spectral_Acceleration/results
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class AcD(torch.optim.Optimizer):
    '''
    此为核心优化算法,即通过一个scaler来加倍步长,三种加倍方式:
    global_scaling 全局加速scaler倍
    flat_scaling_v1 高频速度设为0 低频加速scaler倍
    flat_scaling_v2 高频速度设为0 低频加速scaler倍
    '''
    def __init__(self, params, lr, mode=None, scaler=1.5, momentum=0):
        defaults = dict(lr=lr, mode=mode, scaler=scaler, momentum=momentum)
        super(AcD, self).__init__(params, defaults)
    
    def step(self, flat_matrix=None):
        mode = self.param_groups[0]['mode']
        if mode != 'global_scaling':
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
            grad_flat = flat_matrix(full_grad) 


        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data

                state = self.state[param]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(grad)
                grad = buf

                #print("shape of grad:"+str(grad.shape))
                len_now = len(grad.view(-1))
                # 根据模式修改梯度
                if group['mode'] == 'global_scaling':
                    grad *= group['scaler']
                elif group['mode'] == 'flat_scaling_v1' and flat_matrix is not None:
                    grad_tmp = grad_flat[:len_now]
                    grad_flat = grad_flat[len_now:]
                    grad_tmp = grad_tmp.view_as(grad)
                    grad = grad_tmp* group['scaler'] #这是v1,v1是抛弃高频
                    #grad = grad + (group['scaler']-1)*grad_tmp #这是v2
                elif group['mode'] == 'flat_scaling_v2' and flat_matrix is not None:
                    grad_tmp = grad_flat[:len_now]
                    grad_flat = grad_flat[len_now:]
                    grad_tmp = grad_tmp.view_as(grad)
                    #grad = grad_tmp* group['scaler'] #这是v1
                    grad = grad + (group['scaler']-1)*grad_tmp #这是v2，v2是高频不加速

                param.data.add_(-group['lr'], grad)




def main(dataset: str, arch_id: str, loss: str, opt: str, lr: float, max_steps: int, mode: str="global_scaling",  neigs: int = 0, 
         physical_batch_size: int = 6000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = False,save_middle_model : int = -1, beta: float = 0.0, BS: int = 0,nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 5000, seed: int = 0, scaling: float = 1.0, nfilter: int = 10, cubic:bool=False):
    directory = get_gd_directory(dataset, lr, arch_id, seed, opt, loss, beta, BS)
    if acc_goal == 0.99:
        directory = f"{directory}/acc_0.99/bulk-only"
    else:
        directory = directory
        #directory = f"{directory}/torch_SGD"
    if opt=='polyak' and lr==0.004:
        directory = f"{directory}/warmupbeta_0.95"

    if BS != 0:
        directory = f"{directory}/epoch_{max_steps}"

    
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)
    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)
    train_dataset = [(x.to(device), y.to(device)) for x, y in train_dataset]
    test_dataset = [(x.to(device), y.to(device)) for x, y in test_dataset]
    print(mode)
    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).to(device)


    

    #pretrained_dict = torch.load("/data/users/zhouwenjie/workspace/Spectral_Acceleration/results/cifar10-5k/cnn-relu/seed_0/ce/gd/lr_0.01/acc_0.99/snapshot_1000")
    #pretrained_dict = torch.load(f"{directory}/snapshot_40")
    #network.load_state_dict(pretrained_dict)
    #一些需要加载中间模型的实验
    """
    if not (mode == 'global_scaling' and scaling ==1.0):
        pretrained_dict = torch.load("/data/users/zhouwenjie/workspace/Spectral_Acceleration/results/cifar10/resnet18/seed_0/ce/gd/lr_0.04/BS_1000/epoch_200/snapshot_40")
        #pretrained_dict = torch.load(f"{directory}/snapshot_40")
        network.load_state_dict(pretrained_dict)
    """



    # 初始化一些容器
    len_of_param = len(parameters_to_vector((network.parameters())))
    torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))
    #optimizer = get_gd_optimizer(network.parameters(), opt, lr, beta)
    optimizer = AcD(params=network.parameters(), lr=lr, mode=mode,scaler=scaling, momentum=beta)
    #optimizer = SGD(network.parameters(), lr=lr)
    train_loss, test_loss, train_acc, test_acc = \
        torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    # 现在的实验没有存下eigvecs，因为占空间很大，也很稀疏
    #eigvecs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len_of_param, neigs)
    eigvecs = torch.zeros( len_of_param, neigs)
    gradients = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len_of_param)
    filtered_gradient = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len_of_param)
    Dom_gradient = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len_of_param)
    #fake_sharpness_one = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len_of_param)
    #fake_sharpness_two = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len_of_param)
    
    if cubic == True:
        dom_cubic = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len_of_param)
        bulk_cubic = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len_of_param)
        cubic_term = torch.zeros(max_steps)

    flat_matrix = None
    flag=1

    quadratic_update,Bulk_update, Dom_update = torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    Dom_update[0] = train_loss[0]
    Bulk_update[0] = train_loss[0]
    quadratic_update[0] = train_loss[0]
    step=0

    if BS != 0 :
        while step < max_steps:
            for (X, y) in iterate_dataset(train_dataset, BS):
                train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
                                                                physical_batch_size)
                test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)
                
                # 此为Warmup beta的momentum形式
                if opt=='polyak' and lr==0.004 and step==750:
                    beta = 0.95
                    lr=0.002
                    optimizer = AcD(params=network.parameters(), lr=lr, mode=mode,scaler=scaling, momentum=beta)
                # 此为保存中间模型
                if save_middle_model != -1 and step == save_middle_model:
                    torch.save(network.state_dict(), f"{directory}/snapshot_{step}")

                # 其实是每隔eig_freq步才检查一次flat_matrix，然后再接下来eig_freq步内都使用同一个matrix来过滤
                if eig_freq != -1 and step % eig_freq == 0:

                    # 算三阶的Cubic-vector-vector product
                    if cubic==True:
                        third_order_delta = lambda delta: compute_third_order_hvp(network, loss_fn,abridged_train,
                                                  delta, physical_batch_size=physical_batch_size).detach().cpu()

                    hvp_delta = lambda delta: compute_hvp(network, loss_fn,abridged_train,
                                                delta, physical_batch_size=physical_batch_size).detach().cpu()
                    nparams = len(parameters_to_vector((network.parameters())))
                    eigs[step // eig_freq, :], eigvecs[ :,:] = lanczos(hvp_delta, nparams, neigs=neigs)
                    flat_matrix = compute_flat_matrix(nfilter=nfilter,eigvecs=eigvecs[:,:])

                    gradients[step // eig_freq,:] = compute_gradient(network, loss_fn,[(x.to(device), y.to(device)) for x, y in zip(X,y)])

                    # 如果是GDM，我们这里就把更新量除以学习率的负值作为梯度来存储
                    if opt == 'polyak':
                        if step == 0:
                            gradients[step] =  gradients[step]
                        else:
                            gradients[step] = gradients[step] -beta * gradients[step-1] / lr

                    filtered_gradient[step // eig_freq,:] = flat_matrix(gradients[step // eig_freq,:].to(device))
                    Dom_gradient[step // eig_freq,:] = gradients[step // eig_freq,:] - filtered_gradient[step // eig_freq,:] 

                    if step < max_steps-1:
                        # 算二阶：
                        Dom_update[step+1] = Dom_update[step]- lr * torch.norm(Dom_gradient[step])**2 +\
                                        0.5*lr**2*torch.dot(Dom_gradient[step],hvp_delta(Dom_gradient[step]))
                        quadratic_update[step+1] = quadratic_update[step]- \
                                lr * torch.norm(gradients[step])**2 + 0.5*lr**2*torch.dot(gradients[step],hvp_delta(gradients[step]))
                        Bulk_update[step+1] = Bulk_update[step] - \
                            lr * torch.norm(filtered_gradient[step])**2 + 0.5*lr**2*torch.dot(filtered_gradient[step],hvp_delta(filtered_gradient[step]))
                        
                        # 算三阶：
                        if cubic == True:
                            cubic_term[step+1] = -(1/6) * lr**3 * torch.dot(gradients[step], third_order_delta(gradients[step]))
                            dom_cubic[step+1] =  -(1/6) * lr**3 * torch.dot(Dom_gradient[step], third_order_delta(Dom_gradient[step]))
                            bulk_cubic[step+1] =  -(1/6) * lr**3 * torch.dot(filtered_gradient[step], third_order_delta(filtered_gradient[step]))

                    print("eigenvalues: ", eigs[step//eig_freq, :])

                    # 中途保存
                    if step != 0 and step % (2*eig_freq) == 0 :
                            save_name = "{}_{}_top_{}".format(mode, scaling,nfilter)
                            # 存三阶：
                            if cubic == True:
                                save_files_at_nstep(directory,[("cubic_term", cubic_term[:(step + 1)]),
                                                           ("dom_cubic", dom_cubic[:(step + 1)]),("bulk_cubic", bulk_cubic[:(step + 1)])] , step=save_name)
                            # 存二阶： 
                            save_files_at_nstep(directory,
                            [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                            ("grads", gradients[:(step + 1) // eig_freq]),("filtered_grads", filtered_gradient[:(step + 1) // eig_freq]),
                            ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                            ("dom_update", Dom_update[:step + 1]), ("bulk_update", Bulk_update[:step + 1]),("quadratic_update", quadratic_update[:step + 1]),
                            ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1])], step=save_name)                    
                            
                if iterate_freq != -1 and step % iterate_freq == 0:
                    iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())
                print(f"current:{mode}\t{scaling}\t{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")
                
                if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
                    break


                optimizer.zero_grad()  # 清零梯度
                loss = loss_fn(network(X.to(device)), y.to(device))/ BS  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step(flat_matrix=flat_matrix)  # 更新参数
                step+=1

    else:
        for step in range(max_steps):
            train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
                                                            physical_batch_size)
            test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)
            
            # 此为Warmup beta的momentum形式
            if opt=='polyak' and lr==0.004 and step==750:
                beta = 0.95
                lr=0.002
                optimizer = AcD(params=network.parameters(), lr=lr, mode=mode,scaler=scaling, momentum=beta)
            # 此为保存中间模型
            if save_middle_model != -1 and step == save_middle_model:
                torch.save(network.state_dict(), f"{directory}/snapshot_{step}")

            # 其实是每隔eig_freq步才检查一次flat_matrix，然后再接下来eig_freq步内都使用同一个matrix来过滤
            if eig_freq != -1 and step % eig_freq == 0:

                # 算三阶
                if cubic == True:
                    third_order_delta = lambda delta: compute_third_order_hvp(network, loss_fn,abridged_train,
                                              delta, physical_batch_size=physical_batch_size).detach().cpu()

                hvp_delta = lambda delta: compute_hvp(network, loss_fn,abridged_train,
                                            delta, physical_batch_size=physical_batch_size).detach().cpu()
                nparams = len(parameters_to_vector((network.parameters())))
                eigs[step // eig_freq, :], eigvecs[ :,:] = lanczos(hvp_delta, nparams, neigs=neigs)
                gradients[step // eig_freq,:] = compute_gradient(network,loss_fn, train_dataset)

                # 如果是GDM，我们这里就把更新量除以学习率的负值作为梯度来存储
                if opt == 'polyak':
                    if step == 0:
                        gradients[step] =  gradients[step]
                    else:
                        gradients[step] = gradients[step] -beta * gradients[step-1] / lr

                flat_matrix = compute_flat_matrix(nfilter=nfilter,eigvecs=eigvecs[:,:])
                filtered_gradient[step // eig_freq,:] = flat_matrix(gradients[step // eig_freq,:].to(device))

                Dom_gradient[step // eig_freq,:] = gradients[step // eig_freq,:] - filtered_gradient[step // eig_freq,:] 

                if step < max_steps-1:
                    # 算二阶：
                    Dom_update[step+1] = Dom_update[step]- lr * torch.norm(Dom_gradient[step])**2 +\
                                    0.5*lr**2*torch.dot(Dom_gradient[step],hvp_delta(Dom_gradient[step]))
                    quadratic_update[step+1] = quadratic_update[step]- \
                            lr * torch.norm(gradients[step])**2 + 0.5*lr**2*torch.dot(gradients[step],hvp_delta(gradients[step]))
                    Bulk_update[step+1] = Bulk_update[step] - \
                        lr * torch.norm(filtered_gradient[step])**2 + 0.5*lr**2*torch.dot(filtered_gradient[step],hvp_delta(filtered_gradient[step]))
                    
                    # 算三阶：
                    if cubic == True:
                        cubic_term[step+1] = -(1/6) * lr**3 * torch.dot(gradients[step], third_order_delta(gradients[step]))
                        dom_cubic[step+1] =  -(1/6) * lr**3 * torch.dot(Dom_gradient[step], third_order_delta(Dom_gradient[step]))
                        bulk_cubic[step+1] =  -(1/6) * lr**3 * torch.dot(filtered_gradient[step], third_order_delta(filtered_gradient[step]))

                print("eigenvalues: ", eigs[step//eig_freq, :])

                if step != 0 and step % (2*eig_freq) == 0 :
                        save_name = "{}_{}_top_{}".format(mode, scaling,nfilter)

                        # 存三阶：
                        if cubic == True:
                            save_files_at_nstep(directory,[("cubic_term", cubic_term[:(step + 1)]),
                                                        ("dom_cubic", dom_cubic[:(step + 1)]),("bulk_cubic", bulk_cubic[:(step + 1)])] , step=save_name)
                        
                        # 存二阶： 
                        save_files_at_nstep(directory,
                        [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                        ("grads", gradients[:(step + 1) // eig_freq]),("filtered_grads", filtered_gradient[:(step + 1) // eig_freq]),
                        ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                        ("dom_update", Dom_update[:step + 1]), ("bulk_update", Bulk_update[:step + 1]),("quadratic_update", quadratic_update[:step + 1]),
                        ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1])], step=save_name)                    
                        
            if iterate_freq != -1 and step % iterate_freq == 0:
                iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())
            print(f"current:{mode}\t{scaling}\t{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")
            
            if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
                break

            optimizer.zero_grad()
            for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
                loss = loss_fn(network(X.to(device)), y.to(device)) / len(train_dataset)
                loss.backward()
            optimizer.step(flat_matrix=flat_matrix)
    
    save_name = "{}_{}_top_{}".format(mode, scaling,nfilter)

    
    """
    save_files_at_nstep(directory,
                    [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                    ("grads", gradients[:(step + 1) // eig_freq]),("filtered_grads", filtered_gradient[:(step + 1) // eig_freq]),
                    ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                    ("dom_update", Dom_update[:step + 1]), ("bulk_update", Bulk_update[:step + 1]),("quadratic_update", quadratic_update[:step + 1]),
                    ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1])], step=save_name)    
    """
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_{save_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("lr", type=float, help="the learning rate")
    parser.add_argument("max_steps", type=int, help="the maximum number of gradient steps to train for")
    parser.add_argument("--opt", type=str, choices=["gd", "polyak", "nesterov"],
                        help="which optimization algorithm to use", default="gd")
    parser.add_argument("--mode", type=str, choices=["global_scaling", "flat_scaling_v1", "flat_scaling_v2"],
                        help="which scaling type", default="global_scaling")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--beta", type=float, help="momentum parameter (used if opt = polyak or nesterov)",default=0.0)
    parser.add_argument("--BS", type=int, help="Batchsize(Use it if using SGD)",default=0)
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=6000)
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
    parser.add_argument("--save_middle_model", type=int, default=-1,
                        help="if not -1, save model weights at middle step of training")
    parser.add_argument("--scaling", type=float, default=1.0, help="the scaling")
    parser.add_argument("--nfilter", type=int, default=10, help="the number of top eigenvecter to filter")
    parser.add_argument("--cubic", type=bool, default=False, help="do or do not access cubic taylor expansion")


    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, mode=args.mode, lr=args.lr, max_steps=args.max_steps,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq,
         iterate_freq=args.iterate_freq, save_freq=args.save_freq, save_model=args.save_model,save_middle_model = args.save_middle_model, beta=args.beta,BS = args.BS,
         nproj=args.nproj, loss_goal=args.loss_goal, acc_goal=args.acc_goal, abridged_size=args.abridged_size,
         seed=args.seed, scaling=args.scaling, nfilter=args.nfilter, cubic=args.cubic)