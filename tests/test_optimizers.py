import torch
from synapgrad import Tensor, nn, optim
from utils import check_tensors
import numpy as np


def check_optimizer(opt:optim.Optimizer, param, opt_t:torch.optim.Optimizer, param_t):
    inp = Tensor(np.ones((4,4)), requires_grad=True)
    inp_t = torch.tensor(np.ones((4,4)), requires_grad=True)
    
    print(param)
    
    for _ in range(100):
        # synapgrad
        out = inp @ param
        opt.zero_grad()
        out.sum().backward()
        opt.step()

        # torch   
        out_t = inp_t @ param_t
        opt_t.zero_grad()
        out_t.sum().backward()
        opt_t.step()
        
    print(param)
    print(param_t)
    
    assert check_tensors(inp, inp_t)
    assert check_tensors(param, param_t)
    assert check_tensors(inp.grad, inp_t.grad)
    assert check_tensors(out, out_t)
    assert check_tensors(param, param_t)
    assert check_tensors(param.grad, param_t.grad)


def test_SGD():
    attrs = {
        "lr": 0.1,
        "momentum": 0.9,
        "maximize": False,
        "dampening": 0,
        "nesterov": True,
        "weight_decay": 0.5,
    }
    
    param = Tensor(np.ones((4,4)), requires_grad=True)
    opt = optim.SGD([param], **attrs)
    
    param_t = torch.tensor(np.ones((4,4)), requires_grad=True)
    opt_t = torch.optim.SGD([param_t], **attrs)
    
    check_optimizer(opt, param, opt_t, param_t)
    
    
def test_Adam():
    attrs = {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "weight_decay": 0.7,
        "maximize": False
    }
    
    param = Tensor(np.ones((4,4)), requires_grad=True)
    opt = optim.Adam([param], **attrs)
    
    param_t = torch.tensor(np.ones((4,4)), requires_grad=True)
    opt_t = torch.optim.Adam([param_t], **attrs)
    
    check_optimizer(opt, param, opt_t, param_t)