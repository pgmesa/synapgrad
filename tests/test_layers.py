import torch
from synapgrad import Tensor, nn
from utils import check_tensors
import numpy as np
    
    
def test_linear():
    list_ = [[0.2, 1, 4, 2, -1, 4], [0.2, 1, 4, 2, -1, 4]]
    
    # synapgrad
    inp = Tensor(list_, requires_grad=True)
    linear = nn.Linear(6,3)
    out = linear(inp)
    out = out.sum()
    out.backward()

    # torch
    inp_t = torch.tensor(list_, requires_grad=True)
    linear_t = torch.nn.Linear(6,3)
    out_t = linear_t(inp_t)
    out_t = out_t.sum()
    out_t.backward()

    params = linear.parameters()
    params_t = list(linear_t.parameters())

    assert len(params) == len(params_t)
    for p, p_t in zip(params, params_t):
        assert check_tensors(p.grad, p_t.grad)
        
        
def test_flatten():
    l = np.random.randn(30,28,28,3,4)
    
    # synapgrad
    inp = Tensor(l, requires_grad=True)
    linear = nn.Flatten(start_dim=1, end_dim=2)
    out_l = linear(inp)
    out = out_l.sum()
    out.backward()

    # torch
    inp_t = torch.tensor(l, requires_grad=True)
    linear_t = torch.nn.Flatten(start_dim=1, end_dim=2)
    out_tl = linear_t(inp_t)
    out_t = out_tl.sum()
    out_t.backward()

    assert check_tensors(out_l, out_tl)
    assert check_tensors(inp.grad, inp_t.grad)