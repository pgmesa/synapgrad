import torch
from synapgrad import Tensor, nn
from utils import check_tensors


def test_relu():
    l1 = [[-1.0,-2.0, 4.0, 5.0, 1.0, 7.0], [-1.0,-2.0, 4.0, 5.0, 1.0, 7.0]]
    
    # synapgrad
    a = Tensor(l1, requires_grad=True)
    out = nn.ReLU()(a)
    out = out.sum()
    out.backward()
    
    # torch
    a_t = torch.tensor(l1, requires_grad=True)
    out_t = torch.nn.ReLU()(a_t)
    out_t = out_t.sum()
    out_t.backward()
    
    assert check_tensors(out, out_t)
    assert check_tensors(a.grad, a_t.grad)
    
    
def test_sigmoid():
    l1 = [[-1.0,-2.0, 4.0, 5.0, 1.0, 7.0]]
    
    # synapgrad
    a = Tensor(l1, requires_grad=True)
    b = (a*4)/7 - 2
    out = nn.Sigmoid()(b)*b
    out = out.sum()
    out.backward()
    
    # torch
    a_t = torch.tensor(l1, requires_grad=True)
    b_t = (a_t*4)/7 - 2
    out_t = torch.nn.Sigmoid()(b_t)*b_t
    out_t = out_t.sum()
    out_t.backward()
    
    assert check_tensors(out, out_t)
    assert check_tensors(a.grad, a_t.grad)
    
    
def test_softmax():
    l1 = [[-1.0,-2.0, 4.0, 5.0, 1.0, 7.0]]
    
    # synapgrad
    a = Tensor(l1, requires_grad=True)
    b = (a*4)/7 - 2
    out = nn.Softmax(dim=1)(a)*b
    out = out.sum()
    out.backward()
    
    # torch
    a_t = torch.tensor(l1, requires_grad=True)
    b_t = (a_t*4)/7 - 2
    out_t = torch.nn.Softmax(dim=1)(a_t)*b_t
    out_t = out_t.sum()
    out_t.backward()
    
    assert check_tensors(out, out_t)
    assert check_tensors(a.grad, a_t.grad)