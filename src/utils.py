import torch
import numpy as np


class FakeDL():

    def __init__(self, X, Y, device):
        self.device = device
        self.batchsize = len(X)
        self.X = X.to(device)
        self.Y = Y.to(device)
        self.active = True
        return

    def __len__(self):
        return 1

    def __next__(self):
        if not self.active:
            raise StopIteration
        self.active = False
        return [self.X, self.Y]

    def __iter__(self):
        self.active = True
        return self

    next = __next__


def trace_overlap(s1, s2, device=None):
    M = s1.to(device)
    V = s2.to(device)
    Mt = torch.transpose(M, 0, 1)  # pylint: disable=no-member
    n = M.size()[0]
    k = 0
    for i in range(n):
        vi = V[i]
        li = Mt.mv(M.mv(vi))
        ki = torch.dot(li, li)  # pylint: disable=no-member
        k += ki
    del Mt, M, V
    return float((k / n).cpu().numpy())


def eigenspace_stability(eigenvecs_record):
    n_vecs = len(eigenvecs_record[0])
    steps = len(eigenvecs_record)
    eigenvecs = [[] for i in range(steps)]
    for i in range(steps):
        for j in range(n_vecs):
            v = torch.cat([x.view(-1) for x in eigenvecs_record[i][j]])
            eigenvecs[i].append(v)
    res = np.zeros([n_vecs, steps])
    for i in range(1, steps):
        # top j eigenspace
        for j in range(n_vecs):
            old_space = torch.stack([eigenvecs[i - 1][k] for k in range(j + 1)])
            new_space = torch.stack([eigenvecs[i][k] for k in range(j + 1)])
            ovlp = trace_overlap(old_space, new_space)
            res[j, i] = ovlp
    return res


# Kroneckor Product
def kp_2d(t1, t2):
    t1_h, t1_w = t1.size()
    t2_h, t2_w = t2.size()
    out_h = t1_h * t2_h
    out_w = t1_w * t2_w

    tiled_t2 = t2.repeat(t1_h, t1_w)
    expanded_t1 = (
        t1.unsqueeze(2)
        .unsqueeze(3)
        .repeat(1, t2_h, t2_w, 1)
        .view(out_h, out_w)
    )
    return expanded_t1 * tiled_t2


def bkp_2d_raw(t1, t2):
    btsize, t1_h, t1_w = t1.size()
    btsize, t2_h, t2_w = t2.size()
    out_h = t1_h * t2_h
    out_w = t1_w * t2_w

    tiled_t2 = t2.repeat(1, t1_h, t1_w)
    expanded_t1 = (
        t1.unsqueeze(3)
        .unsqueeze(4)
        .repeat(1, 1, t2_h, t2_w, 1)
        .view(btsize, out_h, out_w)
    )
    expanded_t1 *= tiled_t2
    return expanded_t1


# Batched Kroneckor Product
def bkp_2d(t1, t2):
    btsize_1, t1_h, t1_w = t1.size()
    btsize, t2_h, t2_w = t2.size()
    assert btsize == btsize_1, 'batch size mismatch'
    out_h = t1_h * t2_h
    out_w = t1_w * t2_w

    expanded_t1 = (
        t1.unsqueeze(3)
        .unsqueeze(4)
        .repeat(1, 1, t2_h, t2_w, 1)
        .view(btsize, out_h, out_w)
    )
    for i in range(t1_h):
        for j in range(t1_w):
            expanded_t1[:, i * t2_h: (i + 1) * t2_h, j * t2_w: (j + 1) * t2_w] *= t2
    return expanded_t1
