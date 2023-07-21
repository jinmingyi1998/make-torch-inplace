__all__ = [
    "square_matmul",
    "square_matmul_T",
    "inplace_softmax",
    "inplace_softmax_backward",
    "layernorm",
]

import torch
import make_torch_inplace_C

square_matmul = make_torch_inplace_C.square_matmul
square_matmul_T = make_torch_inplace_C.square_matmul_transposed

inplace_softmax = make_torch_inplace_C.softmax_forward_
inplace_softmax_backward = make_torch_inplace_C.softmax_backward_

layernorm = make_torch_inplace_C.layernorm_
