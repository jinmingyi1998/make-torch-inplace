# Inplace Matmul Softmax LayerNorm

## What is this package doing?

Less memory cost for running forward self-attention based model with torch code

In-pace Matrix multiplication is a tradeoff between memory and time consuming.

## Install

> Supported architectures are specified in [setup.py :: cc_flag](./setup.py#L21)
> [Find "cc" of your CUDA device](https://developer.nvidia.com/cuda-gpus)

```shell
pip install git+https://github.com/jinmingyi1998/make-torch-inplace.git
```

## Usage
### Square matrix multiplication
```python
from make_torch_inplace import square_matmul
a = torch.rand((512,1024)).cuda() 
b = torch.rand((1024,1024)).cuda()
```

square_matmul(Tensor, Tensor, n_rows, n_cols)

```python
a = a.contiguous()
b = b.contiguous()
square_matmul(a,b,512,1024)
```

#### An example for nn.Linear
```python
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
```


### Softmax
```python
from make_torch_inplace import inplace_softmax, inplace_softmax_backward
a = torch.rand((512,1024)).cuda()
```
inplace_softmax(Tensor, n_rows, n_cols)
```python
a = a.contiguous()
inplace_softmax(a,512,1024)
```
also you can `inplace_softmax_backward(something)`

### LayerNorm

This implement doesn't contain element affine. Just calculate x = (x - E(x))/sqrt(V(x)+1e-5) inplace.
So you need apply affine manully
```python
from make_torch_inplace import layernorm as layernorm_C

def layernorm(x:torch.Tensor,m:nn.LayerNorm)->torch.Tensor:
    layernorm_C(x,reduce(mul,x.shape[:-1]),x.shape[-1])
    if m.weight is not None:
        x *= m.weight
    if m.bias is not None:
        x += m.bias
    return x
```