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
    
    
def test_maxpool2d():
    l = np.random.randn(32,5,28,28).astype(np.float32)
    l2 = np.random.randn(32,5,1,1).astype(np.float32)
    
    kernel_size = 5; stride = None; padding = 1; dilation = 2
    
    inp = Tensor(l, requires_grad=True)
    p2 = Tensor(l2)
    pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    out = pool(inp)*p2
    out.sum().backward()
    
    inp_t = torch.tensor(l, requires_grad=True)
    p2_t = torch.tensor(l2)
    pool_t = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    out_t = pool_t(inp_t)*p2_t
    out_t.sum().backward()
    
    print(out.shape, "\n" + str(out)[:100])
    print(out_t.shape, "\n" + str(out_t)[:100])
    
    print(inp.grad.shape, "\n" + str(inp.grad)[:200])
    print(inp_t.grad.shape, "\n" + str(inp_t.grad)[:200])
        
    assert check_tensors(out, out_t)
    assert check_tensors(inp.grad, inp_t.grad)
    

# def test_conv2d():
#     ...
    
    
# def test_batchnorm2d():
#     ...
    

def test_dropout():
    l = np.random.randn(32,5,28,28).astype(np.float32)
    
    inp = Tensor(l, requires_grad=True)
    dropout = nn.Dropout(p=0.3)
    out = dropout(inp)
    out.sum().backward()
    
    dropout.eval()
    out2 = dropout(out)
    
    inp_t = torch.tensor(l, requires_grad=True)
    dropout_t = torch.nn.Dropout(p=0.3)
    out_t = dropout_t(inp_t)
    out_t.sum().backward()
        
    assert out.matches_shape(Tensor(out_t.detach().numpy()))
    assert check_tensors(out, torch.tensor(out2.data))