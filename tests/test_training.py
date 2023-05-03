import torch
from synapgrad import Tensor, nn, optim
from utils import check_tensors
import numpy as np


def test_linear_train():
    model = nn.Linear(3, 3)
    opt = optim.SGD(model.parameters(), lr=0.1)
    
    model_t = torch.nn.Linear(3, 3)
    model_t.weight = torch.nn.parameter.Parameter(torch.tensor(model.weight.data))
    model_t.bias = torch.nn.parameter.Parameter(torch.tensor(model.bias.data))
    opt_t = torch.optim.SGD(model_t.parameters(), lr=0.1)
    
    for _ in range(100):
        l = np.random.randn(3, 3).astype(np.float32)
        inp = Tensor(l, requires_grad=True)
        # synapgrad
        out = model(inp)
        opt.zero_grad()
        out.sum().backward()
        opt.step()

        inp_t = torch.tensor(l, requires_grad=True)
        # torch   
        out_t = model_t(inp_t)
        opt_t.zero_grad()
        out_t.sum().backward()
        opt_t.step()
        
        assert check_tensors(inp, inp_t)
        assert check_tensors(out, out_t)
        assert check_tensors(inp.grad, inp_t.grad)
        assert check_tensors(model.weight, model_t.weight)
        assert check_tensors(model.bias, model_t.bias)
        assert check_tensors(model.weight.grad, model_t.weight.grad)
        assert check_tensors(model.bias.grad, model_t.bias.grad)
    
    
def test_conv2d_train():
    in_channels = 3; out_channels = 5; kernel_size = 3; stride = 1; padding = 'same'
    
    model = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    opt = optim.SGD(model.parameters(), lr=0.1)
    
    model_t = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    model_t.weight = torch.nn.parameter.Parameter(torch.tensor(model.weight.data))
    model_t.bias = torch.nn.parameter.Parameter(torch.tensor(model.bias.data))
    opt_t = torch.optim.SGD(model_t.parameters(), lr=0.1)
    
    for _ in range(100):
        l = np.random.rand(2,in_channels,4,4).astype(np.float32)
        inp = Tensor(l, requires_grad=True)
        # synapgrad
        out = model(inp)
        opt.zero_grad()
        out.sum().backward()
        opt.step()

        inp_t = torch.tensor(l, requires_grad=True)
        # torch   
        out_t = model_t(inp_t)
        opt_t.zero_grad()
        out_t.sum().backward()
        opt_t.step()
    
        assert check_tensors(inp, inp_t)
        assert check_tensors(out, out_t)
        assert check_tensors(inp.grad, inp_t.grad)
        assert check_tensors(model.weight, model_t.weight)
        assert check_tensors(model.bias, model_t.bias)
        assert check_tensors(model.weight.grad, model_t.weight.grad)
        assert check_tensors(model.bias.grad, model_t.bias.grad)
