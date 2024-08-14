import torch
import matplotlib.pyplot as plt
from os import environ,makedirs
import numpy as np

def average_parts(vec, parts):
    averages = []
    start = 0
    for part in parts:
        averages.append(torch.mean(vec[start:start+part]).item())
        start += part
    return averages

def nonzero_ratio(vec, parts):
    ratios = []
    start = 0
    for part in parts:
        # 计算非零数的数量
        nonzero_count = torch.sum((vec[start:start+part] < -1e-5) | (vec[start:start+part] > 1e-5)).item()
        # 计算非零数的比例
        ratio = nonzero_count / part
        ratios.append(ratio)
        start += part
    return ratios

dataset = "cifar10"
arch = "fc-tanh"
loss = "mse"
gd_lr = 0.02
gd_eig_freq = 100
cmap = plt.get_cmap('viridis')
scaling = 1.0
gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"

eigvecs = torch.load(f"{gd_directory}/eigvecs_global_scaling_1.0_top_20_step")
plt.figure(figsize=(10, 10), dpi=100)
print(eigvecs.shape)
l = len(eigvecs)
print(l)
topn=0

parts = [614400, 200, 40000, 200, 2000, 10]

for i in np.linspace(0, l-1, 10, dtype=int):
    eigenvector = eigvecs[i, :, topn]
    # 计算各个部分的平均值
    #averages = average_parts(eigenvector, parts)
    # 创建一个数组，表示每个部分的索引
    #indices = np.arange(len(averages))
    # 使用条形图显示每个部分的平均值
    #plt.bar(indices, averages, color=cmap(i/l), label=f"time={i}")
    ratios = nonzero_ratio(eigenvector, parts)
    indices = np.arange(len(ratios))
    # 使用折线图显示每个部分中非零数的比例
    plt.plot(indices, ratios, color=cmap(i/l), label=f"time={i}")

#bins=len(eigenvector)
plt.xlabel('Part')
plt.ylabel('Nonzero Ratio')
plt.title("eigenvectors")
plt.legend()

makedirs(f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}/figures", exist_ok=True)
plt.savefig(f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}/figures/eigvecs_flow_nonzero.png", bbox_inches='tight', pad_inches=0)
