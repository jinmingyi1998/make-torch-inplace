from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F

from make_torch_inplace import square_matmul, square_matmul_T, layernorm
from time import time_ns

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def inplace_linear(x: torch.Tensor, m: nn.Linear):
    assert (
        m.in_features == m.out_features
    ), "Linear weight should be a square matrix, in_channels should be equal to out_channels"
    assert x.shape[-1] == m.weight.shape[-1], "matmul shape error"
    x = x.contiguous()
    m.weight = m.weight.contiguous()
    square_matmul_T(x, m.weight, reduce(mul, x.shape[:-1]), x.shape[-1])
    if m.bias is not None:
        x += m.bias


torch.backends.cudnn.enabled = False


def test_matmul():
    # with torch.cuda.amp.autocast():

    for i in range(2):
        H = 40000
        W = 1024
        a = torch.randn(H, W).half()
        m = nn.Linear(W, W).half()
        a = a.cuda()
        m = m.cuda()
        t0 = time_ns()
        c = m(a)
        torch.cuda.synchronize()
        t1 = time_ns()
        # print(torch.cuda.memory_summary())
        t2 = time_ns()
        inplace_linear(a, m)
        torch.cuda.synchronize()
        t3 = time_ns()
        print("time(ns)", t1 - t0, t3 - t2)
        d = (a.cpu() - c.cpu()).abs().mean()
        print(d)
        assert d < 0.000001


def torch_impl_layernorm(x):
    m = nn.LayerNorm(x.shape[-1], elementwise_affine=False)
    return m(x)


def my_impl_layernorm(x):
    v = x.var(dim=-1, unbiased=False, keepdim=True)
    e = x.mean(dim=-1, keepdim=True)
    v = v + 1e-5  # same epsilon as torch default
    v = v.sqrt()
    out = (x - e) / v
    return out


def test_layernorm():
    H = 2000
    W = 100
    n_channel = 32
    # a = torch.arange(n_channel).repeat(H * W).reshape((H, W, n_channel)).float().cuda()
    # a = torch.randn(H, W, n_channel).float().cuda()
    # a = a * 1.4
    # a = torch.tensor([[1, 2, 4, 5], [6, 3, 2, 4], [2, 4, 6, 1]]).float().cuda()
    a = torch.arange(H * W * n_channel).reshape((H, W, n_channel)).float()
    # c = F.layer_norm(a, (n_channel,))
    print(a)

    # var mean

    b = my_impl_layernorm(a)

    c = torch_impl_layernorm(a)
    # a = a.contiguous()
    # layernorm(a, reduce(mul, a.shape[:-1]), a.shape[-1])

    print((c - b).abs().mean())
    # print(c)
    # for i in range(c.shape[0]):
    #     if i > 1:
    #         print(c[i] - c[i - 1])
    # print((c - a).abs().mean())
    # print((b - a).abs().mean())
    # print(c)
    # print(a)


def main():
    test_layernorm()


with torch.no_grad():
    if __name__ == "__main__":
        main()
